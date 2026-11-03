# Base file for Julia functions. For backend-specific functionality, see the extension files, e.g. `julia_helper_ext_cuda.jl`.

export to_gpuarray, launch_kernel, synchronize_gpu, metal_create_buffer, metal_create_global_buffer, metal_get_global_buffer, metal_reset_global_buffer, metal_commit_buffer, metal_buffer_is_completed

function to_gpuarray(a)
    a = deepcopy(a) # ensure we have a separate copy of the array to avoid unintended side effects
    if isa(a, AbstractArray)
        a = GPUArrayType(a)
    end
    return a
end

function launch_kernel(kernel, args::Tuple, params::Tuple, ndrange::Tuple, workgroupsize::Tuple, shmem::Int, start_evt::Any, end_evt::Any, stream::Any, use_metal_buffer::Bool)
    launch_time = Inf
    launch_time_start = nothing
    # Check if this is a KernelAbstractions kernel
    if isdefined(Main, :KernelAbstractions) && kt_julia_backend !== nothing && applicable(kernel, kt_julia_backend, workgroupsize)
        configured_kernel = kernel(kt_julia_backend, workgroupsize)
        if use_metal_buffer
            metal_create_global_buffer()
        end
        # Launch kernel asynchronously
        mktemp() do tmppath, _
            open(tmppath, "w") do tmpio
                # kernel errors are printed to stdout, capture them
                redirect_stdout(tmpio) do
                    try
                        val_params = Val.(params)  # convert parameters to Val types for kernel invocation
                        launch_time_start = time_ns()
                        configured_kernel(args..., val_params...; ndrange=ndrange)  # launch the kernel (async)
                        if use_metal_buffer
                            metal_commit_buffer(global_metal_buffer)
                        end
                        launch_time = float((time_ns() - launch_time_start) / 1e6) # convert to milliseconds
                        # Note: kernel launch is asynchronous
                        # Synchronization and event recording is handled by the Python backend
                    catch e
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
    # return launch time; actual timing is done via events in Python backend
    return launch_time, launch_time_start
end

function synchronize_gpu(launch_time_start)
    Main.KernelAbstractions.synchronize(kt_julia_backend)
    if launch_time_start === nothing
        return nothing
    end
    return float((time_ns() - launch_time_start) / 1e6)
end
