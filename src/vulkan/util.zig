const std = @import("std");
const c = @import("c");

pub fn wrapVulkanResult(
    result: c.VkResult,
    comptime message: []const u8,
) error{VulkanInitFailed}!void {
    if (result != c.VK_SUCCESS) {
        std.log.err(message ++ ": {s}", .{c.string_VkResult(result)});
        return error.VulkanInitFailed;
    }
}

pub fn findMemoryTypeIndex(
    required_bits: u32,
    required_prop_flags: u32,
    physical_device: c.VkPhysicalDevice,
) ?u32 {
    var memory_props: c.VkPhysicalDeviceMemoryProperties = undefined;
    c.vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_props);

    for (
        memory_props.memoryTypes[0..memory_props.memoryTypeCount],
        0..,
    ) |memory_type, memory_type_index| {
        if ((required_bits & (@as(u32, 1) << @as(u5, @intCast(memory_type_index))) > 0) and
            ((memory_type.propertyFlags & required_prop_flags) == required_prop_flags))
        {
            return @intCast(memory_type_index);
        }
    }
    return null;
}
