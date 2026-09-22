export to_gpuarray, launch_kernel

function to_gpuarray(a)
    a = deepcopy(a) # ensure we have a separate copy of the array to avoid unintended side effects
    if isa(a, AbstractArray)
        a = GPUArrayType(a)
    end
    return a
end

function launch_kernel(kernel, args::Tuple, params::Tuple, ndrange::Tuple, workgroupsize::Tuple, shmem::Int, start_evt::Any, end_evt::Any, stream::Any)
    launch_time = 0.0
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
                        launch_time_start = time_ns()
                        configured_kernel(args..., val_params...; ndrange=ndrange)  # launch the kernel (async)
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
            # print any stdout output from the kernel
            print(read(tmppath, String))
        end
    else
        error("Only KernelAbstractions kernels are supported.")
    end
    # return launch time; actual timing is done via events in Python backend
    return launch_time
end

function create_metal_buffer(device)
    # Create a Metal buffer in the command queue for timing
    if isdefined(Main, :Metal)
        contextqueue = Main.Metal.MTLCommandQueue(device)
        return Metal.MTLCommandBuffer(contextqueue)
        # return contextqueue.commandBuffer()
    else
        error("Metal backend is not available.")
    end
end
