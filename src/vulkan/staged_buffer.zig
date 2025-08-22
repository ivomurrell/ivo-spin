const c = @import("c");

const Buffer = @import("Buffer.zig");
const util = @import("util.zig");

pub fn StagedBuffer(comptime T: type, buffer_type: u32) type {
    return struct {
        const Self = @This();
        data: []const T,
        handle: Buffer,

        pub fn init(
            data: []const T,
            device: c.VkDevice,
            physical_device: c.VkPhysicalDevice,
            command_pool: c.VkCommandPool,
            graphics_queue: c.VkQueue,
        ) !Self {
            const buffer_size = @sizeOf(T) * data.len;

            const staging_buffer = try Buffer.init(
                device,
                physical_device,
                buffer_size,
                c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            );
            defer staging_buffer.deinit();

            var mapped_data: [*]T = undefined;
            try util.wrapVulkanResult(
                c.vkMapMemory(
                    device,
                    staging_buffer.memory,
                    0,
                    buffer_size,
                    0,
                    @ptrCast(&mapped_data),
                ),
                "failed to map vertex buffer memory",
            );
            @memcpy(mapped_data, data);
            c.vkUnmapMemory(device, staging_buffer.memory);

            const data_buffer = try Buffer.init(
                device,
                physical_device,
                buffer_size,
                c.VK_IMAGE_USAGE_TRANSFER_DST_BIT | buffer_type,
                c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            );
            errdefer data_buffer.deinit();

            const allocate_info = c.VkCommandBufferAllocateInfo{
                .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandPool = command_pool,
                .commandBufferCount = 1,
            };

            var command_buffer: c.VkCommandBuffer = undefined;
            try util.wrapVulkanResult(
                c.vkAllocateCommandBuffers(device, &allocate_info, &command_buffer),
                "failed to allocate copying command buffer",
            );
            defer c.vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);

            const begin_info = c.VkCommandBufferBeginInfo{
                .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };
            try util.wrapVulkanResult(
                c.vkBeginCommandBuffer(command_buffer, &begin_info),
                "failed to begin recording copying command buffer",
            );

            const copy_region = c.VkBufferCopy{
                .size = buffer_size,
            };
            c.vkCmdCopyBuffer(
                command_buffer,
                staging_buffer.handle,
                data_buffer.handle,
                1,
                &copy_region,
            );

            try util.wrapVulkanResult(
                c.vkEndCommandBuffer(command_buffer),
                "failed to end recording copying command buffer",
            );

            const submit_info = c.VkSubmitInfo{
                .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1,
                .pCommandBuffers = &command_buffer,
            };
            try util.wrapVulkanResult(
                c.vkQueueSubmit(graphics_queue, 1, &submit_info, null),
                "failed to submit copying command to queue",
            );
            try util.wrapVulkanResult(
                c.vkQueueWaitIdle(graphics_queue),
                "failed to wait for queue to complete",
            );

            return Self{
                .data = data,
                .handle = data_buffer,
            };
        }

        pub fn deinit(self: Self) void {
            self.handle.deinit();
        }
    };
}
