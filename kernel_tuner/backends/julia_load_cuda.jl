# Makes `CUDACore` and `NVML` available in the module that includes this file.
# These come from the CUDACore.jl and NVML.jl packages if they are installed (direct dependencies of the environment),
# otherwise from CUDA.jl: since v6, CUDA.jl contains CUDACore and NVML, before v6 CUDA.jl itself was the core package.
# This way an existing CUDA.jl installation is used as is, without installing or downloading anything.
if !isdefined(@__MODULE__, :CUDACore)
    if Base.find_package("CUDACore") !== nothing
        import CUDACore
    else
        import CUDA
        const CUDACore = isdefined(CUDA, :CUDACore) ? CUDA.CUDACore : CUDA
    end
end
if !isdefined(@__MODULE__, :NVML)
    if Base.find_package("NVML") !== nothing
        import NVML
    else
        import CUDA
        const NVML = CUDA.NVML
    end
end
