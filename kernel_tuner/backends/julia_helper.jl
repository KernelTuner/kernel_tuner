export to_gpuarray, launch_kernel

function to_gpuarray(a)
    a = deepcopy(a) # ensure we have a separate copy of the array to avoid unintended side effects
    if isa(a, AbstractArray)
        a = GPUArrayType(a)
    end
    return a
end

function launch_kernel(kernel, args::Tuple, params::Tuple, ndrange::Tuple, workgroupsize::Tuple, shmem::Int, start_evt::Any, end_evt::Any, stream::Any)
    t = Inf
    # Check if this is a KernelAbstractions kernel
    if isdefined(Main, :KernelAbstractions) && kt_julia_backend !== nothing && applicable(kernel, kt_julia_backend, workgroupsize)
        configured_kernel = kernel(kt_julia_backend, workgroupsize)
        # Launch kernel
        mktemp() do tmppath, _
            open(tmppath, "w") do tmpio
                # kernel errors are printed to stdout, capture them
                redirect_stdout(tmpio) do
                    try
                        val_params = Val.(params)  # convert parameters to Val types for kernel invocation
                        start = time_ns()   # simple host-side timing as fallback in case of issues with GPU timing
                        if start_evt !== nothing
                            if isdefined(Main, :CUDA) && isa(start_evt, CuEvent)
                                Main.CUDA.record(start_evt, stream)
                            elseif isdefined(Main, :AMDGPU) && isa(start_evt, HIPEvent)
                                Main.AMDGPU.HIP.record(start_evt)
                            else
                                error("Unsupported event type for timing: $(typeof(start_evt))")
                            end
                        end
                        configured_kernel(args..., val_params...; ndrange=ndrange)  # launch the kernel
                        Main.KernelAbstractions.synchronize(kt_julia_backend) # synchronize to ensure kernel completion
                        if end_evt !== nothing
                            if isdefined(Main, :CUDA) && isa(end_evt, CuEvent)
                                Main.CUDA.record(end_evt, stream)
                                Main.CUDA.synchronize(end_evt) # ensure the event is recorded before we read it
                            elseif isdefined(Main, :AMDGPU) && isa(end_evt, HIPEvent)
                                Main.AMDGPU.HIP.record(end_evt)
                                Main.AMDGPU.HIP.synchronize(end_evt) # ensure the event is recorded before we read it
                            else
                                error("Unsupported event type for timing: $(typeof(end_evt))")
                            end
                        end
                        t = float((time_ns() - start) / 1e6) # convert to milliseconds
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
            # print any stdout output from the kernel
            print(read(tmppath, String))
        end
    else
        error("Only KernelAbstractions kernels are supported.")
    end
    return t
end
