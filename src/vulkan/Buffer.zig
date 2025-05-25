const c = @import("../c.zig");

const util = @import("util.zig");

const Buffer = @This();

device: c.VkDevice,
handle: c.VkBuffer,
memory: c.VkDeviceMemory,

pub fn init(
    device: c.VkDevice,
    physical_device: c.VkPhysicalDevice,
    buffer_size: u64,
    usage_flags: u32,
    required_prop_flags: u32,
) !Buffer {
    const buffer_create_info = c.VkBufferCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = buffer_size,
        .usage = usage_flags,
        .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
    };
    var buffer: c.VkBuffer = undefined;
    try util.wrapVulkanResult(
        c.vkCreateBuffer(device, &buffer_create_info, null, &buffer),
        "failed to create vertex buffer",
    );
    errdefer c.vkDestroyBuffer(device, buffer, null);

    var mem_reqs: c.VkMemoryRequirements = undefined;
    c.vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);
    const memory_type_index =
        util.findMemoryTypeIndex(
            mem_reqs.memoryTypeBits,
            required_prop_flags,
            physical_device,
        ) orelse return error.VulkanInitFailed;

    const allocate_info = c.VkMemoryAllocateInfo{
        .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = memory_type_index,
    };
    var buffer_memory: c.VkDeviceMemory = undefined;
    try util.wrapVulkanResult(
        c.vkAllocateMemory(device, &allocate_info, null, &buffer_memory),
        "failed to allocate vertex buffer memory",
    );
    errdefer c.vkFreeMemory(device, buffer_memory, null);

    try util.wrapVulkanResult(
        c.vkBindBufferMemory(device, buffer, buffer_memory, 0),
        "failed to bind vertx buffer memory",
    );

    return Buffer{
        .device = device,
        .handle = buffer,
        .memory = buffer_memory,
    };
}

pub fn deinit(self: Buffer) void {
    c.vkDestroyBuffer(self.device, self.handle, null);
    c.vkFreeMemory(self.device, self.memory, null);
}
