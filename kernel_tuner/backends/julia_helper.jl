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

    function launch_kernel(kernel, args::Tuple, grid::NTuple{3,Int}, block::NTuple{3,Int}, shmem::Int)
        # Check if this is a KernelAbstractions kernel
        if isdefined(Main, :KernelAbstractions) && backend !== nothing && applicable(kernel, backend, block)
            # Calculate ndrange from grid and block
            workgroupsize = block
            ndrange = (grid[1] * block[1], grid[2] * block[2], grid[3] * block[3])
            # Launch kernel
            configured_kernel = kernel(backend, workgroupsize)
            configured_kernel(args..., ndrange=ndrange)
            # Synchronize to ensure kernel completion
            CUDA.synchronize()
        else
            warn("KernelAbstractions not found, falling back to standard CUDA.jl kernel launch.")
            # Standard CUDA.jl kernel (KernelAbstractions not loaded)
            CUDA.@sync @cuda threads=block blocks=grid shmem=shmem kernel(args...)
        end
    end
end
