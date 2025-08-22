const c = @import("c");

const util = @import("util.zig");

pub fn Image(
    format: c.VkFormat,
    tiling: c.VkImageTiling,
    usage: c.VkImageUsageFlags,
    mem_properties: c.VkMemoryPropertyFlags,
) type {
    return struct {
        const Self = @This();
        device: c.VkDevice,
        handle: c.VkImage,
        memory: c.VkDeviceMemory,
        format: c.VkFormat,

        pub fn init(
            device: c.VkDevice,
            physical_device: c.VkPhysicalDevice,
            width: u32,
            height: u32,
        ) !Self {
            const image_create_info = c.VkImageCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType = c.VK_IMAGE_TYPE_2D,
                .format = format,
                .extent = c.VkExtent3D{
                    .width = width,
                    .height = height,
                    .depth = 1,
                },
                .mipLevels = 1,
                .arrayLayers = 1,
                .tiling = tiling,
                .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
                .usage = usage,
                .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
                .samples = c.VK_SAMPLE_COUNT_1_BIT,
            };
            var image: c.VkImage = undefined;
            try util.wrapVulkanResult(
                c.vkCreateImage(device, &image_create_info, null, &image),
                "failed to create image",
            );
            errdefer c.vkDestroyImage(device, image, null);

            var mem_reqs: c.VkMemoryRequirements = undefined;
            c.vkGetImageMemoryRequirements(device, image, &mem_reqs);
            const memory_type_index = util.findMemoryTypeIndex(
                mem_reqs.memoryTypeBits,
                mem_properties,
                physical_device,
            ) orelse return error.VulkanInitFailed;
            const allocate_info = c.VkMemoryAllocateInfo{
                .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = mem_reqs.size,
                .memoryTypeIndex = memory_type_index,
            };
            var image_memory: c.VkDeviceMemory = undefined;
            try util.wrapVulkanResult(
                c.vkAllocateMemory(device, &allocate_info, null, &image_memory),
                "failed to allocate image memory",
            );
            errdefer c.vkFreeMemory(device, image_memory, null);
            try util.wrapVulkanResult(
                c.vkBindImageMemory(device, image, image_memory, 0),
                "failed to bind image memory",
            );

            return Self{
                .device = device,
                .handle = image,
                .memory = image_memory,
                .format = format,
            };
        }

        pub fn deinit(self: Self) void {
            c.vkDestroyImage(self.device, self.handle, null);
            c.vkFreeMemory(self.device, self.memory, null);
        }
    };
}
