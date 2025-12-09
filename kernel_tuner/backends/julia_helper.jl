module KernelTunerHelper
    using CUDA
    const backend = CUDABackend()   # currently only CUDA backend is supported

    export to_cuarray, launch_kernel

    function to_cuarray(x)
        if isa(x, CuArray)
            return x
        elseif isa(x, AbstractArray)
            return CuArray(x)
        else
            return x
        end
    end

    function launch_kernel(kernel, args::Tuple, params::Tuple, ndrange::Tuple, workgroupsize::Tuple, shmem::Int)
        # Check if this is a KernelAbstractions kernel
        if isdefined(Main, :KernelAbstractions) && backend !== nothing && applicable(kernel, backend, workgroupsize)
            # Launch kernel
            configured_kernel = kernel(backend, workgroupsize)
            configured_kernel(args..., Val.(params)..., ndrange=ndrange)
            # Synchronize to ensure kernel completion
            CUDA.synchronize()
        else
            error("Currently, only KernelAbstractions kernels are supported.")
        end
    end
end
