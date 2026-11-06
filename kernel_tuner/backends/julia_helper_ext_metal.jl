# Extension of the julia_helper.jl file for the Metal backend. You can assume `using Metal` has ran.

# GPU timing. Metal.jl batches kernel launches into shared command buffers, which are only committed later, so a kernel
# launch has no command buffer of its own to time. Instead, the command buffers that Metal.jl commits while launching are
# recorded through its profiling hook, and the batch is committed right after the launch so that the kernel starts running.
# Kernel Tuner synchronizes before every launch, so the recorded command buffers only contain this kernel.
const metal_launched_buffers = Any[]

function metal_launch_recorded(launch)
    empty!(metal_launched_buffers)
    previous_hook = Metal.MTL.profile_hook[]
    Metal.MTL.profile_hook[] = function (cmdbuf)
        push!(metal_launched_buffers, cmdbuf)
        previous_hook === nothing || previous_hook(cmdbuf)
    end
    try
        launch()
        Metal.flush!()  # commit the batch that holds the kernel
    finally
        Metal.MTL.profile_hook[] = previous_hook
    end
end

metal_is_done(cmdbuf::Metal.MTL.MTLCommandBuffer) = cmdbuf.status >= Metal.MTL.MTLCommandBufferStatusCompleted  # also when it failed
metal_is_done(submission) = Metal.MTL.is_completed(submission)  # Metal 4 submissions

# whether the command buffers of the last launch have finished
metal_kernel_finished() = all(metal_is_done, metal_launched_buffers)

# the GPU time in milliseconds of the last launch, from the start of its first command buffer to the end of its last,
# or `nothing` if not available (e.g. not finished yet)
function metal_gpu_time_ms()
    isempty(metal_launched_buffers) && return nothing
    ranges = [Metal.MTL.gpu_time_range(cmdbuf) for cmdbuf in metal_launched_buffers]
    any(isnothing, ranges) && return nothing
    return (maximum(last, ranges) - minimum(first, ranges)) * 1e3
end
