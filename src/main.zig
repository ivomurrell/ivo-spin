const std = @import("std");
const c = @import("c.zig");
const SDL = @import("sdl.zig");
const Vulkan = @import("vulkan.zig");

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const sdl = try SDL.init();
    defer sdl.deinit();

    var vulkan = try Vulkan.init(allocator, &sdl);
    defer vulkan.deinit();

    el: while (true) {
        var event = c.SDL_Event{ .type = 0 };
        while (c.SDL_PollEvent(&event)) {
            if (event.type == c.SDL_EVENT_QUIT) {
                break :el;
            }
        }
        try vulkan.drawFrame();
    }
}
