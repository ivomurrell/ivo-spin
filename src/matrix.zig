const std = @import("std");

pub const Vec3 = @Vector(3, f32);
pub const Vec4 = @Vector(4, f32);

pub const Mat4 = extern struct {
    const Self = @This();
    x: Vec4,
    y: Vec4,
    z: Vec4,
    t: Vec4,

    pub fn identity() Self {
        return Self{
            .x = .{ 1, 0, 0, 0 },
            .y = .{ 0, 1, 0, 0 },
            .z = .{ 0, 0, 1, 0 },
            .t = .{ 0, 0, 0, 1 },
        };
    }

    pub fn translate(x: f32, y: f32, z: f32) Self {
        return Self{
            .x = .{ 1, 0, 0, 0 },
            .y = .{ 0, 1, 0, 0 },
            .z = .{ 0, 0, 1, 0 },
            .t = .{ x, y, z, 1 },
        };
    }

    pub fn rotateX(angle: f32) Self {
        return Self{
            .x = .{ 1, 0, 0, 0 },
            .y = .{ 0, @cos(angle), @sin(angle), 0 },
            .z = .{ 0, -@sin(angle), @cos(angle), 0 },
            .t = .{ 0, 0, 0, 1 },
        };
    }

    pub fn rotateY(angle: f32) Self {
        return Self{
            .x = .{ @cos(angle), 0, -@sin(angle), 0 },
            .y = .{ 0, 1, 0, 0 },
            .z = .{ @sin(angle), 0, @cos(angle), 0 },
            .t = .{ 0, 0, 0, 1 },
        };
    }

    pub fn rotateZ(angle: f32) Self {
        return Self{
            .x = .{ @cos(angle), @sin(angle), 0, 0 },
            .y = .{ -@sin(angle), @cos(angle), 0, 0 },
            .z = .{ 0, 0, 1, 0 },
            .t = .{ 0, 0, 0, 1 },
        };
    }

    pub fn rotate(x: f32, y: f32, z: f32) Self {
        return rotateX(x).mult(rotateY(y)).mult(rotateZ(z));
    }

    pub fn lookAt(eye: Vec3, centre: Vec3, up: Vec3) Self {
        const z = centre - eye;
        const z_n = normalise(3, z);
        const up_n = normalise(3, up);
        const x = cross(z_n, up_n);
        const x_n = normalise(3, x);
        const y = cross(x_n, z_n);
        const y_n = normalise(3, y);

        return Self{
            .x = .{ x_n[0], x_n[1], x_n[2], 0 },
            .y = .{ y_n[0], y_n[1], y_n[2], 0 },
            .z = .{ z_n[0], z_n[1], z_n[2], 0 },
            .t = .{ -dot(3, x_n, eye), -dot(3, y_n, eye), -dot(3, z_n, eye), 1 },
        };
    }

    pub fn perspective(fov: f32, aspect_ratio: f32, near: f32, far: f32) Self {
        const f = 1 / @tan(fov / 2);
        return Self{
            .x = .{ f / aspect_ratio, 0, 0, 0 },
            .y = .{ 0, -f, 0, 0 },
            .z = .{ 0, 0, far / (far - near), 1 },
            .t = .{ 0, 0, (near * far) / (near - far), 0 },
        };
    }

    pub fn transpose(self: Self) Self {
        const T = @typeInfo(Self);
        var transposed: Self = undefined;
        inline for (T.Struct.fields, 0..) |field, col| {
            @field(transposed, field.name) = .{
                self.x[col],
                self.y[col],
                self.z[col],
                self.t[col],
            };
        }
        return transposed;
    }

    pub fn mult(left: Self, right: Self) Self {
        const left_t = left.transpose();
        return Self{
            .x = .{
                dot(4, left_t.x, right.x),
                dot(4, left_t.y, right.x),
                dot(4, left_t.z, right.x),
                dot(4, left_t.t, right.x),
            },
            .y = .{
                dot(4, left_t.x, right.y),
                dot(4, left_t.y, right.y),
                dot(4, left_t.z, right.y),
                dot(4, left_t.t, right.y),
            },
            .z = .{
                dot(4, left_t.x, right.z),
                dot(4, left_t.y, right.z),
                dot(4, left_t.z, right.z),
                dot(4, left_t.t, right.z),
            },
            .t = .{
                dot(4, left_t.x, right.t),
                dot(4, left_t.y, right.t),
                dot(4, left_t.z, right.t),
                dot(4, left_t.t, right.t),
            },
        };
    }
};

fn expectApproxEqualVec(expected: Vec4, actual: Vec4) !void {
    const V = @typeInfo(Vec4).Vector;
    const tolerance = comptime std.math.floatEps(V.child);

    for (0..V.len) |i| {
        std.testing.expectApproxEqAbs(expected[i], actual[i], tolerance) catch |err| {
            std.debug.print(
                "index {d} incorrect. expected {d}, found {d}\n",
                .{ i, expected[i], actual[i] },
            );
            return err;
        };
    }
}

fn expectApproxEqualMatrix(expected: Mat4, actual: Mat4) !void {
    inline for (@typeInfo(Mat4).Struct.fields) |field| {
        const expected_row = @field(expected, field.name);
        const actual_row = @field(actual, field.name);
        expectApproxEqualVec(expected_row, actual_row) catch |err| {
            std.debug.print(
                "field {s} incorrect. expected {any}, found {any}\n",
                .{ field.name, expected_row, actual_row },
            );
            return err;
        };
    }
}

pub fn magnitude(comptime len: comptime_int, vec: @Vector(len, f32)) f32 {
    const squared = vec * vec;
    return @sqrt(@reduce(std.builtin.ReduceOp.Add, squared));
}

pub fn normalise(comptime len: comptime_int, vec: @Vector(len, f32)) @Vector(len, f32) {
    const mag = magnitude(len, vec);
    return vec / @as(@Vector(len, f32), @splat(mag));
}

pub fn dot(comptime len: comptime_int, left: @Vector(len, f32), right: @Vector(len, f32)) f32 {
    const product = left * right;
    return @reduce(std.builtin.ReduceOp.Add, product);
}

test dot {
    try std.testing.expectEqual(38, dot(4, .{ 0, 1, 2, 3 }, .{ 4, 5, 6, 7 }));
}

pub fn cross(left: Vec3, right: Vec3) Vec3 {
    return .{
        left[1] * right[2] - left[2] * right[1],
        left[0] * right[2] - left[2] * right[0],
        left[0] * right[1] - left[1] * right[0],
    };
}

pub fn vertexMult(mat: Mat4, vertex: Vec3) Vec4 {
    const homog = Vec4{ vertex[0], vertex[1], vertex[2], 1 };
    const mat_t = mat.transpose();
    return Vec4{
        dot(4, mat_t.x, homog),
        dot(4, mat_t.y, homog),
        dot(4, mat_t.z, homog),
        dot(4, mat_t.t, homog),
    };
}

test "Mat4.translate" {
    try expectApproxEqualMatrix(Mat4.identity(), Mat4.translate(0, 0, 0));
    try expectApproxEqualVec(
        Vec4{ -9, 8, 56, 1 },
        vertexMult(Mat4.translate(-10, 6, 53), .{ 1, 2, 3 }),
    );
}

test "Mat4.rotate" {
    try expectApproxEqualMatrix(Mat4.identity(), Mat4.rotate(0, 0, 0));
    try expectApproxEqualVec(
        Vec4{ 0, -1, 0, 1 },
        vertexMult(
            Mat4.rotate(std.math.degreesToRadians(90), std.math.degreesToRadians(180), 0),
            .{ 0, 0, -1 },
        ),
    );
}

test "Mat4.lookAt" {
    try expectApproxEqualMatrix(
        Mat4.translate(0, 0, 0.5),
        Mat4.lookAt(.{ 0, 0, -0.5 }, .{ 0, 0, 0 }, .{ 0, -1, 0 }),
    );
    try expectApproxEqualMatrix(
        Mat4.rotateY(std.math.pi).mult(Mat4.translate(0, 0, -0.5)),
        Mat4.lookAt(.{ 0, 0, 0.5 }, .{ 0, 0, 0 }, .{ 0, -1, 0 }),
    );
}

test "Mat4.transpose" {
    const mat = Mat4{
        .x = .{ 1, 2, 3, 0 },
        .y = .{ 4, 5, 6, 0 },
        .z = .{ 7, 8, 9, 0 },
        .t = .{ -1, -2, -3, 1 },
    };
    try expectApproxEqualMatrix(Mat4{
        .x = .{ 1, 4, 7, -1 },
        .y = .{ 2, 5, 8, -2 },
        .z = .{ 3, 6, 9, -3 },
        .t = .{ 0, 0, 0, 1 },
    }, mat.transpose());
}

test "Mat4.mult" {
    const left = Mat4{
        .x = .{ 1, 2, 3, 0 },
        .y = .{ 4, 5, 6, 0 },
        .z = .{ 7, 8, 9, 0 },
        .t = .{ -1, -2, -3, 1 },
    };
    const right = Mat4{
        .x = .{ 8, 2, 5, 0 },
        .y = .{ 0, 8, 6, 0 },
        .z = .{ 7, 4, 3, 0 },
        .t = .{ -4, -5, -6, 1 },
    };
    try expectApproxEqualMatrix(Mat4{
        .x = .{ 51, 66, 81, 0 },
        .y = .{ 74, 88, 102, 0 },
        .z = .{ 44, 58, 72, 0 },
        .t = .{ -67, -83, -99, 1 },
    }, left.mult(right));
}
