# Base file for Julia functions. For backend-specific functionality, see the extension files, e.g. `julia_helper_ext_cuda.jl`.

export to_gpuarray, launch_kernel, synchronize_gpu, metal_kernel_finished, metal_gpu_time_ms

# GPU timing events recorded directly around a kernel launch, specialized in the backend extension files.
# By default (backends without timing events), there are no events.
create_timing_events(backend) = (nothing, nothing)
record_timing_event(::Nothing) = nothing

function to_gpuarray(a)
    # ensure we have a separate copy to avoid unintended side effects; for arrays, the conversion to a GPU array
    # already copies, while a deepcopy of a view would needlessly copy its entire parent array
    # (on the CPU backend, `Array(a)` returns `a` itself for an `Array`, in which case it is copied explicitly)
    isa(a, AbstractArray) || return deepcopy(a)
    b = GPUArrayType(a)
    return b === a ? deepcopy(a) : b
end

function launch_kernel(kernel, args::Tuple, params::Tuple, ndrange::Tuple, workgroupsize::Tuple, shmem::Int, use_metal_buffer::Bool)
    launch_time = Inf
    launch_time_start = nothing
    start_evt = nothing
    end_evt = nothing
    # Check if this is a KernelAbstractions kernel
    if isdefined(Main, :KernelAbstractions) && kt_julia_backend !== nothing && applicable(kernel, kt_julia_backend, workgroupsize)
        configured_kernel = kernel(kt_julia_backend, workgroupsize)
        # Launch kernel asynchronously
        mktemp() do tmppath, _
            open(tmppath, "w") do tmpio
                # kernel errors are printed to stdout, capture them
                redirect_stdout(tmpio) do
                    try
                        val_params = Val.(params)  # convert parameters to Val types for kernel invocation
                        start_evt, end_evt = create_timing_events(kt_julia_backend)
                        # Record the timing events directly around the launch, so that host-side work before the
                        # launch is not measured as GPU time. The GC is disabled meanwhile, as a GC pause between
                        # recording the start event and launching would otherwise be measured as GPU time.
                        gc_was_enabled = GC.enable(false)
                        try
                            launch_time_start = time_ns()
                            record_timing_event(start_evt)
                            if use_metal_buffer
                                # record the command buffers of this launch, used for GPU timing (see julia_helper_ext_metal.jl)
                                metal_launch_recorded(() -> configured_kernel(args..., val_params...; ndrange=ndrange))
                            else
                                configured_kernel(args..., val_params...; ndrange=ndrange)  # launch the kernel (async)
                            end
                            record_timing_event(end_evt)
                            launch_time = float((time_ns() - launch_time_start) / 1e6) # convert to milliseconds
                        finally
                            GC.enable(gc_was_enabled)
                        end
                        # Note: kernel launch is asynchronous
                        # Synchronization and event recording is handled by the Python backend
                    catch e
                        e isa InterruptException && rethrow()  # Ctrl-C: Kernel Tuner stops tuning (see `util.is_interrupt`)
                        redirect_stdout(stdout) # restore stdout
                        close(tmpio)
                        stdout_output = read(tmppath, String)
                        print("Kernel stdout during exception:\n", stdout_output)
                        # Rethrow the exception to be caught outside
                        throw(stdout_output * "\n" * sprint(showerror, e, catch_backtrace()))
                    end
                end
            end
            # # print any stdout output from the kernel (for debugging only, disabled for performance)
            # print(read(tmppath, String))
        end
    else
        error("Only KernelAbstractions kernels are supported.")
    end
    # return the launch time and the GPU timing events (if any), which are used by the observer in the Python backend
    return launch_time, launch_time_start, start_evt, end_evt
end

function synchronize_gpu(launch_time_start)
    Main.KernelAbstractions.synchronize(kt_julia_backend)
    if launch_time_start === nothing
        return nothing
    end
    return float((time_ns() - launch_time_start) / 1e6)
end
