const c = @import("c");

const Buffer = @import("buffer.zig");
const util = @import("util.zig");

const Vertex = @This();

position: @Vector(3, f32),
colour: @Vector(3, f32),

pub fn bindingDescription() c.VkVertexInputBindingDescription {
    return .{
        .binding = 0,
        .stride = @sizeOf(Vertex),
        .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX,
    };
}

pub fn attributeDescriptions() [2]c.VkVertexInputAttributeDescription {
    return [_]c.VkVertexInputAttributeDescription{
        .{
            .binding = 0,
            .location = 0,
            .format = c.VK_FORMAT_R32G32B32_SFLOAT,
            .offset = @offsetOf(Vertex, "position"),
        },
        .{
            .binding = 0,
            .location = 1,
            .format = c.VK_FORMAT_R32G32B32_SFLOAT,
            .offset = @offsetOf(Vertex, "colour"),
        },
    };
}
