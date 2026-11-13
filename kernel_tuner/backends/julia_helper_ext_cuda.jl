# Extension of the julia_helper.jl file for the CUDA backend. You can assume `using CUDACore` has ran.
# GPU timing events, recorded on the stream that the kernel is launched on
create_timing_events(::CUDABackend) = (CuEvent(), CuEvent())
record_timing_event(evt::CuEvent) = CUDACore.record(evt, CUDACore.stream())
