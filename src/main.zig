const std = @import("std");
const c = @import("c");

const Input = @import("Input.zig");
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

    var input: Input = .init;

    el: while (true) {
        var event = c.SDL_Event{ .type = 0 };
        while (c.SDL_PollEvent(&event)) {
            switch (event.type) {
                c.SDL_EVENT_QUIT => {
                    break :el;
                },
                c.SDL_EVENT_KEY_UP, c.SDL_EVENT_KEY_DOWN => {
                    input.updateInput(event.key);
                },
                else => {},
            }
        }
        try vulkan.drawFrame(input);
    }
}
