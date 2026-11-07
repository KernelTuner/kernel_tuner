# Extension of the julia_helper.jl file for the AMDGPU backend. You can assume `using AMDGPU` has ran.
# GPU timing events, recorded on the stream that the kernel is launched on
create_timing_events(::ROCBackend) = (
    AMDGPU.HIP.HIPEvent(AMDGPU.stream(); do_record=false, timing=true),
    AMDGPU.HIP.HIPEvent(AMDGPU.stream(); do_record=false, timing=true),
)
record_timing_event(evt::AMDGPU.HIP.HIPEvent) = AMDGPU.HIP.record(evt)
