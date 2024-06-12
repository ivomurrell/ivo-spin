const std = @import("std");
const c = @import("c.zig");

const Self = @This();

window: *c.SDL_Window,

pub fn init() !Self {
    if (c.SDL_Init(c.SDL_INIT_VIDEO) < 0) {
        std.log.err(
            "Couldn't initialize SDL: {s}",
            .{c.SDL_GetError()},
        );
        return error.SDLInitFailed;
    }

    const window = c.SDL_CreateWindow("ivo spinny", 640, 480, c.SDL_WINDOW_VULKAN) orelse {
        std.log.err(
            "Couldn't create window: {s}",
            .{c.SDL_GetError()},
        );
        return error.SDLInitFailed;
    };

    return Self{ .window = window };
}

pub fn deinit(self: Self) void {
    c.SDL_DestroyWindow(self.window);
    c.SDL_Quit();
}

pub fn extensions() []const [*:0]const u8 {
    var extension_count: u32 = 0;
    const extension_list = c.SDL_Vulkan_GetInstanceExtensions(&extension_count);
    return @ptrCast(extension_list[0..extension_count]);
}

pub fn initVulkanSurface(self: *const Self, instance: c.VkInstance) !c.VkSurfaceKHR {
    var surface: c.VkSurfaceKHR = null;
    if (c.SDL_Vulkan_CreateSurface(self.window, instance, null, &surface) != c.SDL_TRUE) {
        return error.SDLInitSurfaceFailed;
    }
    return surface;
}

pub fn deinitSurface(instance: c.VkInstance, surface: c.VkSurfaceKHR) void {
    c.SDL_Vulkan_DestroySurface(instance, surface, null);
}
