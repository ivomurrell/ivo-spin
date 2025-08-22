const std = @import("std");
const Allocator = std.mem.Allocator;
const Endian = std.builtin.Endian;

const Vertex = @import("vulkan/Vertex.zig");
const mat = @import("matrix.zig");

const Gltf = @This();

const Buffer = struct {
    byteLength: u32,
};
const BufferView = struct {
    buffer: u16,
    byteOffset: u32 = 0,
    byteLength: u32,
    byteStride: ?u32 = null,
};
const Node = struct {
    children: ?[]u16 = null,
    mesh: ?u16 = null,
    matrix: ?[16]f32 = null,
};
const MeshPrimitive = struct {
    attributes: std.json.Value,
    indices: ?u16 = null,
};
const Mesh = struct {
    primitives: []MeshPrimitive,
};
const ComponentType = enum(u16) {
    byte = 5120,
    ubyte,
    short,
    ushort,
    uint = 5125,
    float,
};
const Accessor = struct {
    bufferView: u16,
    byteOffset: u32 = 0,
    componentType: ComponentType,
    count: u32,
    type: []u8,
};
const Json = struct {
    buffers: []Buffer,
    bufferViews: []BufferView,
    nodes: []Node,
    meshes: []Mesh,
    accessors: []Accessor,
};

const GltfError = error{
    FileTooShort,
    InvalidFileFormat,
    CorruptData,
    InvalidModel,
};

allocator: Allocator,
json: std.json.Parsed(Json),
bin_offset: u32,
file: std.fs.File,

pub fn parse(allocator: Allocator, path: []const u8) !Gltf {
    const file = try std.fs.cwd().openFile(path, .{});
    errdefer file.close();
    var file_reader = file.reader(&.{});

    var header: [20]u8 = undefined;
    try file_reader.interface.readSliceAll(&header);
    if (!std.mem.eql(u8, header[0..4], "glTF")) {
        return GltfError.InvalidFileFormat;
    }
    if (std.mem.readInt(u32, header[4..8], Endian.little) != 2) {
        return GltfError.InvalidFileFormat;
    }
    if (!std.mem.eql(u8, header[16..20], "JSON")) {
        return GltfError.InvalidFileFormat;
    }

    const json_len = std.mem.readInt(u32, header[12..16], Endian.little);

    var json_buffer: [4096]u8 = undefined;
    var limited_reader = file_reader.interface.limited(
        .limited(json_len),
        &json_buffer,
    );
    var json_reader: std.json.Reader = .init(allocator, &limited_reader.interface);
    defer json_reader.deinit();

    const json = try (std.json.parseFromTokenSource(
        Json,
        allocator,
        &json_reader,
        .{ .ignore_unknown_fields = true },
    ) catch GltfError.CorruptData);
    errdefer json.deinit();

    var bin_header: [8]u8 = undefined;
    try file_reader.interface.readSliceAll(&bin_header);
    const bin_len = std.mem.readInt(u32, bin_header[0..4], Endian.little);
    const actual_bin_len = try file.getEndPos() - file_reader.logicalPos();
    if (bin_len != actual_bin_len) {
        return GltfError.CorruptData;
    }
    if (!std.mem.eql(u8, bin_header[4..8], "BIN\x00")) {
        return GltfError.InvalidFileFormat;
    }

    return .{
        .allocator = allocator,
        .json = json,
        .bin_offset = @intCast(file_reader.logicalPos()),
        .file = file,
    };
}

pub fn deinit(self: Gltf) void {
    self.json.deinit();
    self.file.close();
}

pub const Model = struct {
    const Self = @This();

    vertices: std.ArrayListUnmanaged([3]f32),
    indices: std.ArrayListUnmanaged(u32),

    pub const empty: Self = .{ .vertices = .empty, .indices = .empty };

    pub fn deinit(self: *Self, allocator: Allocator) void {
        self.vertices.deinit(allocator);
        self.indices.deinit(allocator);
    }
};

pub fn loadModel(self: Gltf) !Model {
    var model: Model = .empty;
    try self.loadNode(&model, 0);
    return model;
}

fn loadNode(self: Gltf, model: *Model, node_index: u16) !void {
    const json = self.json.value;

    const node = json.nodes[node_index];
    if (node.children) |children| {
        for (children) |child| {
            const vertex_child_start = model.vertices.items.len;
            try self.loadNode(model, child);
            if (node.matrix) |matrix_flat| {
                const matrix = mat.Mat4{
                    .x = matrix_flat[0..4].*,
                    .y = matrix_flat[4..8].*,
                    .z = matrix_flat[8..12].*,
                    .t = matrix_flat[12..16].*,
                };
                const new_vertices = model.vertices.items[vertex_child_start..];
                for (new_vertices) |*vertex| {
                    vertex.* = @as(
                        [4]f32,
                        mat.vertexMult(matrix, vertex.*),
                    )[0..3].*;
                }
            }
        }
    } else if (node.mesh) |mesh_index| {
        const mesh = json.meshes[mesh_index];
        const primitive = mesh.primitives[0];

        const indices_accessor_index = primitive.indices orelse return GltfError.InvalidModel;
        const indices_data = try self.readFromAccessor(
            u32,
            &model.indices,
            indices_accessor_index,
        );
        for (indices_data) |*index| {
            index.* += @intCast(model.vertices.items.len);
        }

        const vertex_accessor_index = (primitive.attributes.object
            .get("POSITION") orelse return GltfError.InvalidModel).integer;
        _ = try self.readFromAccessor(
            [3]f32,
            &model.vertices,
            @intCast(vertex_accessor_index),
        );
    }
}

pub fn readFromAccessor(
    self: Gltf,
    comptime T: type,
    buf: *std.ArrayListUnmanaged(T),
    index: usize,
) ![]T {
    const json = self.json.value;

    const accessor = json.accessors[index];
    const buffer_view = json.bufferViews[accessor.bufferView];

    var file_reader = self.file.reader(&.{});
    try file_reader.seekTo(
        self.bin_offset + buffer_view.byteOffset + accessor.byteOffset,
    );
    const data = try buf.addManyAsSlice(self.allocator, accessor.count);
    try file_reader.interface.readSliceAll(@ptrCast(data));
    return data;
}
