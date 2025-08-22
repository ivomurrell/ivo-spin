const c = @import("c");

const mat = @import("../matrix.zig");

const Buffer = @import("Buffer.zig");
const util = @import("util.zig");

pub fn UniformBuffer(comptime T: type) type {
    return struct {
        const Self = @This();
        handle: Buffer,
        mapped_memory: *T,

        pub fn init(device: c.VkDevice, physical_device: c.VkPhysicalDevice) !Self {
            const buffer_size = @sizeOf(T);
            const buffer = try Buffer.init(
                device,
                physical_device,
                buffer_size,
                c.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            );
            errdefer buffer.deinit();

            var mapped_memory: *T = undefined;
            try util.wrapVulkanResult(
                c.vkMapMemory(device, buffer.memory, 0, buffer_size, 0, @ptrCast(&mapped_memory)),
                "failed to map uniform buffer memory",
            );

            return Self{
                .handle = buffer,
                .mapped_memory = mapped_memory,
            };
        }

        pub fn deinit(self: Self) void {
            self.handle.deinit();
        }
    };
}
