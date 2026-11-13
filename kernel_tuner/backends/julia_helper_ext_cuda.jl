# Extension of the julia_helper.jl file for the CUDA backend. You can assume `using CUDACore` has ran.
# Only use exported names unqualified, as CUDACore may be provided by CUDA.jl (see julia_load_cuda.jl).
# GPU timing events, recorded on the stream that the kernel is launched on
create_timing_events(::CUDABackend) = (CuEvent(), CuEvent())
record_timing_event(evt::CuEvent) = record(evt, stream())
