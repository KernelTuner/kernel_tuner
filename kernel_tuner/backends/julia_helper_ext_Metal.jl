# Extension of the julia_helper.jl file for Metal backend, to avoid polluting the main julia_helper.jl file with Metal-specific code and allow Metal-specific types

function metal_create_buffer()::Metal.MTL.MTLCommandBuffer
    queue = Metal.global_queue(Metal.device())
    return Metal.MTLCommandBuffer(queue)
end

function metal_create_global_buffer()::Metal.MTL.MTLCommandBuffer
    queue = Metal.global_queue(Metal.device())
    global global_metal_buffer = Metal.MTLCommandBuffer(queue)
    return global_metal_buffer
end

function metal_get_global_buffer()::Metal.MTL.MTLCommandBuffer
    return global_metal_buffer
end

function metal_reset_global_buffer()
    global global_metal_buffer = nothing
end

function metal_commit_buffer()
    # Commit the default Metal command buffer to ensure it is executed
    return Metal.commit!(global_metal_buffer)
end

function metal_commit_buffer(buffer::Metal.MTL.MTLCommandBuffer)
    # Commit the Metal command buffer to ensure it is executed
    return Metal.commit!(buffer)
end

function metal_buffer_is_completed()
    # Check if the default Metal command buffer has completed execution
    if global_metal_buffer == nothing
        return false    # might result in deadlock, but prevents queued kernels from terminating early (e.g. for continous observer)
    end
    return global_metal_buffer.status >= Metal.MTL.MTLCommandBufferStatusCompleted   # this also captures MTLCommandBufferStatusError as it has a higher value
end

function metal_buffer_is_completed(buffer::Metal.MTL.MTLCommandBuffer)
    # Check if the Metal command buffer has completed execution
    return buffer.status >= Metal.MTL.MTLCommandBufferStatusCompleted   # this also captures MTLCommandBufferStatusError as it has a higher value
end