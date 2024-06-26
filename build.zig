const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) !void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const shader_path = "src/shaders/";
    var shader_dir = try std.fs.cwd().openDir(shader_path, .{ .iterate = true });
    defer shader_dir.close();
    var walker = try shader_dir.walk(b.allocator);
    const allowed_exts = [_][]const u8{ ".vert", ".frag" };
    const Shader = struct { output: std.Build.LazyPath, filename: []const u8 };
    var shader_outputs = std.ArrayList(Shader).init(b.allocator);
    while (try walker.next()) |entry| {
        const ext = std.fs.path.extension(entry.basename);
        const should_include_file = for (allowed_exts) |allowed_ext| {
            if (std.mem.eql(u8, ext, allowed_ext))
                break true;
        } else false;
        if (should_include_file) {
            const shader_compile = b.addSystemCommand(&.{"glslc"});
            const input_path = try std.fs.path.join(
                b.allocator,
                &[_][]const u8{ shader_path, entry.path },
            );
            shader_compile.addFileArg(b.path(input_path));
            const out_path = b.fmt("{s}.spv", .{ext[1..]});
            const spv_out = shader_compile.addPrefixedOutputFileArg("-o", out_path);
            try shader_outputs.append(.{ .output = spv_out, .filename = out_path });
        }
    }

    const exe = b.addExecutable(.{
        .name = "ivo-spin",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.linkSystemLibrary("sdl3");
    if (target.query.isNative() and target.result.os.tag == .macos) {
        // the vulkan utility headers aren't declared in a pkg-config file so
        // we need to add the brew include path
        exe.addIncludePath(std.Build.LazyPath{ .cwd_relative = "/opt/homebrew/include" });
        // the vulkan loader can't find the validation layer unless we add its
        // path to the search paths here
        exe.addLibraryPath(std.Build.LazyPath{ .cwd_relative = "/opt/homebrew/lib" });
    }
    exe.linkSystemLibrary("vulkan");
    for (shader_outputs.items) |shader_output| {
        exe.root_module.addAnonymousImport(
            shader_output.filename,
            .{ .root_source_file = shader_output.output },
        );
    }

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);

    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = b.addRunArtifact(exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}
