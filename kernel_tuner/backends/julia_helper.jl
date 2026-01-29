export to_gpuarray, launch_kernel

function to_gpuarray(a)
    if isa(a, AbstractArray)
        a = gpu_array_type(a)
    end
    return a
end

function launch_kernel(kernel, args::Tuple, params::Tuple, ndrange::Tuple, workgroupsize::Tuple, shmem::Int)
    # Check if this is a KernelAbstractions kernel
    if isdefined(Main, :KernelAbstractions) && kt_julia_backend !== nothing && applicable(kernel, kt_julia_backend, workgroupsize)
        configured_kernel = kernel(kt_julia_backend, workgroupsize)
        # Launch kernel
        mktemp() do tmppath, _
            open(tmppath, "w") do tmpio
                # kernel errors are printed to stdout, capture them
                redirect_stdout(tmpio) do
                    try
                        configured_kernel(args..., Val.(params)..., ndrange=ndrange)
                        # Synchronize to ensure kernel completion
                        Main.KernelAbstractions.synchronize(kt_julia_backend)
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
end
