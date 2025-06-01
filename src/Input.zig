const c = @import("c.zig");

const Input = @This();

up: bool,
down: bool,
left: bool,
right: bool,

pub const init: Input = .{
    .up = false,
    .down = false,
    .left = false,
    .right = false,
};

pub fn updateInput(self: *Input, key: c.SDL_KeyboardEvent) void {
    const input_field = switch (key.scancode) {
        c.SDL_SCANCODE_UP => &self.up,
        c.SDL_SCANCODE_DOWN => &self.down,
        c.SDL_SCANCODE_LEFT => &self.left,
        c.SDL_SCANCODE_RIGHT => &self.right,
        else => null,
    };
    if (input_field) |arrow_field| {
        arrow_field.* = key.down;
    }
}

pub fn isUp(self: Input) bool {
    return self.up and !self.down;
}
pub fn isDown(self: Input) bool {
    return self.down and !self.up;
}
pub fn isLeft(self: Input) bool {
    return self.left and !self.right;
}
pub fn isRight(self: Input) bool {
    return self.right and !self.left;
}
