export to_gpuarray, launch_kernel

function to_gpuarray(x)
    if isa(x, GPUArrayType)
        return x
    elseif isa(x, AbstractArray)
        return GPUArrayType(x)
    else
        return x
    end
end

function launch_kernel(kernel, args::Tuple, params::Tuple, ndrange::Tuple, workgroupsize::Tuple, shmem::Int)
    # Check if this is a KernelAbstractions kernel
    if isdefined(Main, :KernelAbstractions) && kt_julia_backend !== nothing && applicable(kernel, kt_julia_backend, workgroupsize)
        # Launch kernel
        configured_kernel = kernel(kt_julia_backend, workgroupsize)
        configured_kernel(args..., Val.(params)..., ndrange=ndrange)
        # Synchronize to ensure kernel completion
        Main.KernelAbstractions.synchronize(kt_julia_backend)
    else
        error("Only KernelAbstractions kernels are supported.")
    end
end
