const std = @import("std");
const builtin = @import("builtin");
const c = @import("c.zig");
const SDL = @import("sdl.zig");

const Allocator = std.mem.Allocator;

const Self = @This();

allocator: Allocator,
instance: c.VkInstance,
surface: c.VkSurfaceKHR,
device: c.VkDevice,
graphics_queue: c.VkQueue,
present_queue: c.VkQueue,
swapchain: c.VkSwapchainKHR,
swapchain_images: []c.VkImage,
swapchain_image_format: c.VkFormat,
swapchain_extent: c.VkExtent2D,
swapchain_image_views: []c.VkImageView,
render_pass: c.VkRenderPass,
pipeline_layout: c.VkPipelineLayout,
graphics_pipeline: c.VkPipeline,
framebuffers: []c.VkFramebuffer,
command_pool: c.VkCommandPool,
command_buffer: c.VkCommandBuffer,
image_available_semaphore: c.VkSemaphore,
render_finished_semaphore: c.VkSemaphore,
in_flight_fence: c.VkFence,

pub fn init(allocator: Allocator, sdl: *const SDL) !Self {
    const instance = try initInstance(allocator);
    errdefer c.vkDestroyInstance(instance, null);

    const surface = try sdl.initVulkanSurface(instance);
    errdefer SDL.deinitSurface(instance, surface);

    const physical_device = try initPhysicalDevice(instance);
    var device_properties: c.VkPhysicalDeviceProperties = undefined;
    c.vkGetPhysicalDeviceProperties(physical_device, &device_properties);
    std.log.debug("selected {s} device", .{device_properties.deviceName});

    const queue_family_indices = try getQueueFamilyIndices(
        allocator,
        physical_device,
        surface,
        &device_properties.deviceName,
    );

    const device = try initDevice(physical_device, queue_family_indices);
    errdefer c.vkDestroyDevice(device, null);

    var graphics_queue: c.VkQueue = undefined;
    c.vkGetDeviceQueue(device, queue_family_indices.graphics_family, 0, &graphics_queue);
    var present_queue: c.VkQueue = undefined;
    c.vkGetDeviceQueue(device, queue_family_indices.present_family, 0, &present_queue);

    const swapchain = try initSwapchain(
        allocator,
        surface,
        physical_device,
        device,
        queue_family_indices,
    );
    errdefer {
        c.vkDestroySwapchainKHR(device, swapchain.handle, null);
        allocator.free(swapchain.images);
    }

    const swapchain_image_views = try initImageViews(
        allocator,
        device,
        swapchain.images,
        swapchain.image_format,
    );
    errdefer {
        for (swapchain_image_views) |image_view| {
            c.vkDestroyImageView(device, image_view, null);
        }
        allocator.free(swapchain_image_views);
    }

    const render_pass = try initRenderPass(device, swapchain.image_format);
    errdefer c.vkDestroyRenderPass(device, render_pass, null);

    const pipeline_layout = try initPipelineLayout(device);
    errdefer c.vkDestroyPipelineLayout(device, pipeline_layout, null);

    const graphics_pipeline = try initGraphicsPipeline(
        device,
        swapchain.image_extent,
        pipeline_layout,
        render_pass,
    );
    errdefer c.vkDestroyPipeline(device, graphics_pipeline, null);

    const framebuffers = try initFramebuffers(
        allocator,
        device,
        swapchain.image_extent,
        swapchain_image_views,
        render_pass,
    );
    errdefer {
        for (framebuffers) |framebuffer| {
            c.vkDestroyFramebuffer(device, framebuffer, null);
        }
        allocator.free(framebuffers);
    }

    const command_pool = try initCommandPool(device, queue_family_indices);
    errdefer c.vkDestroyCommandPool(device, command_pool, null);

    const command_buffer = try initCommandBuffer(device, command_pool);

    const synchronisation_objects = try initSynchronisation(device);
    errdefer {
        c.vkDestroySemaphore(device, synchronisation_objects.image_available_semaphore, null);
        c.vkDestroySemaphore(device, synchronisation_objects.render_finished_semaphore, null);
        c.vkDestroyFence(device, synchronisation_objects.in_flight_fence, null);
    }

    return Self{
        .allocator = allocator,
        .instance = instance,
        .surface = surface,
        .device = device,
        .graphics_queue = graphics_queue,
        .present_queue = present_queue,
        .swapchain = swapchain.handle,
        .swapchain_images = swapchain.images,
        .swapchain_image_format = swapchain.image_format,
        .swapchain_extent = swapchain.image_extent,
        .swapchain_image_views = swapchain_image_views,
        .render_pass = render_pass,
        .pipeline_layout = pipeline_layout,
        .graphics_pipeline = graphics_pipeline,
        .framebuffers = framebuffers,
        .command_pool = command_pool,
        .command_buffer = command_buffer,
        .image_available_semaphore = synchronisation_objects.image_available_semaphore,
        .render_finished_semaphore = synchronisation_objects.render_finished_semaphore,
        .in_flight_fence = synchronisation_objects.in_flight_fence,
    };
}

pub fn deinit(self: Self) void {
    wrapVulkanResult(
        c.vkDeviceWaitIdle(self.device),
        "couldn't wait for device to be finished",
    ) catch {};

    c.vkDestroySemaphore(self.device, self.image_available_semaphore, null);
    c.vkDestroySemaphore(self.device, self.render_finished_semaphore, null);
    c.vkDestroyFence(self.device, self.in_flight_fence, null);
    c.vkDestroyCommandPool(self.device, self.command_pool, null);
    for (self.framebuffers) |framebuffer| {
        c.vkDestroyFramebuffer(self.device, framebuffer, null);
    }
    self.allocator.free(self.framebuffers);
    c.vkDestroyPipeline(self.device, self.graphics_pipeline, null);
    c.vkDestroyPipelineLayout(self.device, self.pipeline_layout, null);
    c.vkDestroyRenderPass(self.device, self.render_pass, null);
    for (self.swapchain_image_views) |image_view| {
        c.vkDestroyImageView(self.device, image_view, null);
    }
    self.allocator.free(self.swapchain_image_views);
    self.allocator.free(self.swapchain_images);
    c.vkDestroySwapchainKHR(self.device, self.swapchain, null);
    c.vkDestroyDevice(self.device, null);
    SDL.deinitSurface(self.instance, self.surface);
    c.vkDestroyInstance(self.instance, null);
}

fn wrapVulkanResult(
    result: c.VkResult,
    comptime message: []const u8,
) error{VulkanInitFailed}!void {
    if (result != c.VK_SUCCESS) {
        std.log.err(message ++ ": {s}", .{c.string_VkResult(result)});
        return error.VulkanInitFailed;
    }
}

fn initInstance(allocator: Allocator) !c.VkInstance {
    const sdl_extensions = SDL.extensions();
    const portability_extensions = [_][*:0]const u8{
        c.VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        c.VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
    };
    var extensions = try allocator.alloc(
        [*:0]const u8,
        sdl_extensions.len + portability_extensions.len,
    );
    defer allocator.free(extensions);
    @memcpy(extensions[0..sdl_extensions.len], sdl_extensions);
    @memcpy(
        extensions[sdl_extensions.len..][0..portability_extensions.len],
        &portability_extensions,
    );

    var create_info = c.VkInstanceCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .flags = c.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
        .enabledExtensionCount = @intCast(extensions.len),
        .ppEnabledExtensionNames = @ptrCast(extensions),
    };
    if (builtin.mode == std.builtin.OptimizeMode.Debug) {
        const validation_layers = [_][]const u8{"VK_LAYER_KHRONOS_validation"};
        create_info.enabledLayerCount = validation_layers.len;
        create_info.ppEnabledLayerNames = @ptrCast(&validation_layers);
    }

    var instance: c.VkInstance = undefined;
    try wrapVulkanResult(
        c.vkCreateInstance(&create_info, null, &instance),
        "vulkan instance failed to initialise",
    );

    return instance;
}

fn initPhysicalDevice(instance: c.VkInstance) !c.VkPhysicalDevice {
    var physical_device_count: u32 = 1;
    var physical_device: c.VkPhysicalDevice = undefined;
    try wrapVulkanResult(
        c.vkEnumeratePhysicalDevices(instance, &physical_device_count, &physical_device),
        "failed to find vulkan device",
    );
    return physical_device;
}

const QueueFamilyIndices = struct {
    graphics_family: u32,
    present_family: u32,
};

fn getQueueFamilyIndices(
    allocator: Allocator,
    physical_device: c.VkPhysicalDevice,
    surface: c.VkSurfaceKHR,
    device_name: []const u8,
) !QueueFamilyIndices {
    var queue_family_count: u32 = 0;
    c.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, null);
    const queue_families = try allocator.alloc(c.VkQueueFamilyProperties, queue_family_count);
    defer allocator.free(queue_families);
    c.vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device,
        &queue_family_count,
        queue_families.ptr,
    );

    var graphics_family: ?u32 = null;
    var present_family: ?u32 = null;
    return for (queue_families, 0..) |queue_family, i| {
        const queue_family_index: u32 = @intCast(i);

        if (graphics_family == null and
            (queue_family.queueFlags & c.VK_QUEUE_GRAPHICS_BIT) != 0)
        {
            graphics_family = queue_family_index;
        }
        if (present_family == null) {
            var physical_support: c.VkBool32 = undefined;
            _ = c.vkGetPhysicalDeviceSurfaceSupportKHR(
                physical_device,
                queue_family_index,
                surface,
                &physical_support,
            );
            if (physical_support == c.VK_TRUE) {
                present_family = queue_family_index;
            }
        }

        for ([_]?u32{ graphics_family, present_family }) |family_index| {
            if (family_index == null) {
                break;
            }
        } else {
            return QueueFamilyIndices{
                .graphics_family = graphics_family.?,
                .present_family = present_family.?,
            };
        }
    } else {
        std.log.err(
            "failed to find required vulkan queue families for device {s}",
            .{device_name},
        );
        return error.VulkanInitFailed;
    };
}

fn initDevice(
    physical_device: c.VkPhysicalDevice,
    queue_family_indices: QueueFamilyIndices,
) !c.VkDevice {
    const queue_create_indices = [_]u32{
        queue_family_indices.graphics_family,
        queue_family_indices.present_family,
    };
    var queue_create_infos = [_]c.VkDeviceQueueCreateInfo{c.VkDeviceQueueCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueCount = 1,
        .pQueuePriorities = &@as(f32, 1.0),
    }} ** queue_create_indices.len;
    for (&queue_create_infos, queue_create_indices) |*queue_create_info, queue_create_index| {
        queue_create_info.*.queueFamilyIndex = queue_create_index;
    }
    const queue_create_info_unique_count: u32 =
        if (queue_family_indices.graphics_family == queue_family_indices.present_family) 1 else 2;

    const device_features = c.VkPhysicalDeviceFeatures{};
    const device_extensions = [_][*c]const u8{
        c.VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
        c.VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };
    const create_info = c.VkDeviceCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pQueueCreateInfos = &queue_create_infos,
        .queueCreateInfoCount = queue_create_info_unique_count,
        .pEnabledFeatures = &device_features,
        .enabledExtensionCount = device_extensions.len,
        .ppEnabledExtensionNames = &device_extensions,
    };
    var device: c.VkDevice = undefined;
    try wrapVulkanResult(
        c.vkCreateDevice(physical_device, &create_info, null, &device),
        "failed to create logical device",
    );
    return device;
}

const Swapchain = struct {
    handle: c.VkSwapchainKHR,
    images: []c.VkImage,
    image_format: c.VkFormat,
    image_extent: c.VkExtent2D,
};

fn initSwapchain(
    allocator: Allocator,
    surface: c.VkSurfaceKHR,
    physical_device: c.VkPhysicalDevice,
    device: c.VkDevice,
    queue_family_indices: QueueFamilyIndices,
) !Swapchain {
    var surface_capabilities: c.VkSurfaceCapabilitiesKHR = undefined;
    try wrapVulkanResult(c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        physical_device,
        surface,
        &surface_capabilities,
    ), "failed to get surface capabilities");

    const ideal_image_count = surface_capabilities.minImageCount + 1;
    const image_count = @min(ideal_image_count, surface_capabilities.maxImageCount);

    const image_format = c.VK_FORMAT_B8G8R8A8_SRGB;
    const image_extent = surface_capabilities.currentExtent;

    var create_info = c.VkSwapchainCreateInfoKHR{
        .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = image_count,
        .imageFormat = image_format,
        .imageColorSpace = c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        .imageExtent = image_extent,
        .imageArrayLayers = 1,
        .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .preTransform = surface_capabilities.currentTransform,
        .compositeAlpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = c.VK_PRESENT_MODE_FIFO_KHR,
        .clipped = c.VK_TRUE,
    };
    if (queue_family_indices.graphics_family != queue_family_indices.present_family) {
        create_info.imageSharingMode = c.VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = 2;
        create_info.pQueueFamilyIndices = &[_]u32{
            queue_family_indices.graphics_family,
            queue_family_indices.present_family,
        };
    } else {
        create_info.imageSharingMode = c.VK_SHARING_MODE_EXCLUSIVE;
    }

    var swapchain: c.VkSwapchainKHR = undefined;
    try wrapVulkanResult(
        c.vkCreateSwapchainKHR(device, &create_info, null, &swapchain),
        "failed to create swapchain",
    );

    var swapchain_image_count: u32 = undefined;
    try wrapVulkanResult(
        c.vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, null),
        "failed to get number of swapchain images",
    );
    const swapchain_images = try allocator.alloc(c.VkImage, swapchain_image_count);
    errdefer allocator.free(swapchain_images);
    try wrapVulkanResult(c.vkGetSwapchainImagesKHR(
        device,
        swapchain,
        &swapchain_image_count,
        swapchain_images.ptr,
    ), "failed to get number of swapchain images");

    return Swapchain{
        .handle = swapchain,
        .images = swapchain_images,
        .image_format = image_format,
        .image_extent = image_extent,
    };
}

fn initImageViews(
    allocator: Allocator,
    device: c.VkDevice,
    images: []c.VkImage,
    image_format: c.VkFormat,
) ![]c.VkImageView {
    const image_views = try allocator.alloc(c.VkImageView, images.len);
    errdefer allocator.free(image_views);

    var num_created: usize = 0;
    errdefer for (image_views[0..num_created]) |image_view| {
        c.vkDestroyImageView(device, image_view, null);
    };
    for (images, image_views) |image, *imageView| {
        var create_info = c.VkImageViewCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
            .format = image_format,
            .subresourceRange = c.VkImageSubresourceRange{
                .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        try wrapVulkanResult(
            c.vkCreateImageView(device, &create_info, null, imageView),
            "failed to create an image view",
        );
        num_created += 1;
    }

    return image_views;
}

fn initRenderPass(device: c.VkDevice, imageFormat: c.VkFormat) !c.VkRenderPass {
    const colour_attachment = c.VkAttachmentDescription{
        .format = imageFormat,
        .samples = c.VK_SAMPLE_COUNT_1_BIT,
        .loadOp = c.VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = c.VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = c.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = c.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };
    const colour_attachment_ref = c.VkAttachmentReference{
        .attachment = 0,
        .layout = c.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    const subpass = c.VkSubpassDescription{
        .pipelineBindPoint = c.VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colour_attachment_ref,
    };
    const subpass_dependency = c.VkSubpassDependency{
        .srcSubpass = c.VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = c.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };

    const create_info = c.VkRenderPassCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &colour_attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &subpass_dependency,
    };
    var render_pass: c.VkRenderPass = undefined;
    try wrapVulkanResult(
        c.vkCreateRenderPass(device, &create_info, null, &render_pass),
        "failed to create render pass",
    );
    return render_pass;
}

fn initPipelineLayout(device: c.VkDevice) !c.VkPipelineLayout {
    const create_info = c.VkPipelineLayoutCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    };
    var pipeline_layout: c.VkPipelineLayout = undefined;
    try wrapVulkanResult(c.vkCreatePipelineLayout(
        device,
        &create_info,
        null,
        &pipeline_layout,
    ), "failed to create pipeline layout");
    return pipeline_layout;
}

fn initGraphicsPipeline(
    device: c.VkDevice,
    extent: c.VkExtent2D,
    pipeline_layout: c.VkPipelineLayout,
    render_pass: c.VkRenderPass,
) !c.VkPipeline {
    const vert_shader_code align(@alignOf(u32)) = @embedFile("vert.spv").*;
    const frag_shader_code align(@alignOf(u32)) = @embedFile("frag.spv").*;

    const vert_shader = try create_shader_module(device, &vert_shader_code);
    defer c.vkDestroyShaderModule(device, vert_shader, null);
    const frag_shader = try create_shader_module(device, &frag_shader_code);
    defer c.vkDestroyShaderModule(device, frag_shader, null);

    const vert_stage_create_info = c.VkPipelineShaderStageCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = c.VK_SHADER_STAGE_VERTEX_BIT,
        .module = vert_shader,
        .pName = "main",
    };
    const frag_stage_create_info = c.VkPipelineShaderStageCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = c.VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = frag_shader,
        .pName = "main",
    };
    const shader_stages = [_]c.VkPipelineShaderStageCreateInfo{
        vert_stage_create_info,
        frag_stage_create_info,
    };

    const vertex_input_info = c.VkPipelineVertexInputStateCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 0,
        .vertexAttributeDescriptionCount = 0,
    };
    const input_assembly_info = c.VkPipelineInputAssemblyStateCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = c.VK_FALSE,
    };
    const viewport_state_info = c.VkPipelineViewportStateCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &c.VkViewport{
            .x = 0.0,
            .y = 0.0,
            .width = @floatFromInt(extent.width),
            .height = @floatFromInt(extent.height),
            .minDepth = 0.0,
            .maxDepth = 1.0,
        },
        .scissorCount = 1,
        .pScissors = &c.VkRect2D{ .offset = .{ .x = 0, .y = 0 }, .extent = extent },
    };
    const rasteriser_state_info = c.VkPipelineRasterizationStateCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = c.VK_FALSE,
        .rasterizerDiscardEnable = c.VK_FALSE,
        .polygonMode = c.VK_POLYGON_MODE_FILL,
        .lineWidth = 1.0,
        .cullMode = c.VK_CULL_MODE_BACK_BIT,
        .frontFace = c.VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = c.VK_FALSE,
    };
    const multisample_state_info = c.VkPipelineMultisampleStateCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .sampleShadingEnable = c.VK_FALSE,
        .rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT,
    };
    const colour_blend_state_info = c.VkPipelineColorBlendStateCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = c.VK_FALSE,
        .attachmentCount = 1,
        .pAttachments = &c.VkPipelineColorBlendAttachmentState{
            .colorWriteMask = c.VK_COLOR_COMPONENT_R_BIT |
                c.VK_COLOR_COMPONENT_G_BIT |
                c.VK_COLOR_COMPONENT_B_BIT |
                c.VK_COLOR_COMPONENT_A_BIT,
            .blendEnable = c.VK_FALSE,
        },
    };

    const create_info = c.VkGraphicsPipelineCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = &shader_stages,
        .pVertexInputState = &vertex_input_info,
        .pInputAssemblyState = &input_assembly_info,
        .pViewportState = &viewport_state_info,
        .pRasterizationState = &rasteriser_state_info,
        .pMultisampleState = &multisample_state_info,
        .pColorBlendState = &colour_blend_state_info,
        .layout = pipeline_layout,
        .renderPass = render_pass,
        .subpass = 0,
    };
    var graphics_pipeline: c.VkPipeline = undefined;
    try wrapVulkanResult(c.vkCreateGraphicsPipelines(
        device,
        null,
        1,
        &create_info,
        null,
        &graphics_pipeline,
    ), "failed to create graphics pipeline");
    return graphics_pipeline;
}

fn create_shader_module(device: c.VkDevice, bytecode: []align(4) const u8) !c.VkShaderModule {
    const create_info = c.VkShaderModuleCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = bytecode.len,
        .pCode = @ptrCast(bytecode.ptr),
    };
    var shader_module: c.VkShaderModule = undefined;
    try wrapVulkanResult(
        c.vkCreateShaderModule(device, &create_info, null, &shader_module),
        "failed to create shader module",
    );
    return shader_module;
}

fn initFramebuffers(
    allocator: Allocator,
    device: c.VkDevice,
    image_extent: c.VkExtent2D,
    image_views: []c.VkImageView,
    render_pass: c.VkRenderPass,
) ![]c.VkFramebuffer {
    const framebuffers = try allocator.alloc(c.VkFramebuffer, image_views.len);

    var num_created: usize = 0;
    errdefer for (framebuffers[0..num_created]) |framebuffer| {
        c.vkDestroyFramebuffer(device, framebuffer, null);
    };
    for (image_views, framebuffers) |image_view, *framebuffer| {
        const create_info = c.VkFramebufferCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = render_pass,
            .attachmentCount = 1,
            .pAttachments = &image_view,
            .width = image_extent.width,
            .height = image_extent.height,
            .layers = 1,
        };
        try wrapVulkanResult(
            c.vkCreateFramebuffer(device, &create_info, null, framebuffer),
            "failed to create framebuffer",
        );
        num_created += 1;
    }

    return framebuffers;
}

fn initCommandPool(device: c.VkDevice, queue_family_indices: QueueFamilyIndices) !c.VkCommandPool {
    const create_info = c.VkCommandPoolCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queue_family_indices.graphics_family,
    };
    var command_pool: c.VkCommandPool = undefined;
    try wrapVulkanResult(
        c.vkCreateCommandPool(device, &create_info, null, &command_pool),
        "failed to create command pool",
    );
    return command_pool;
}

fn initCommandBuffer(device: c.VkDevice, command_pool: c.VkCommandPool) !c.VkCommandBuffer {
    const allocate_info = c.VkCommandBufferAllocateInfo{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    var command_buffer: c.VkCommandBuffer = undefined;
    try wrapVulkanResult(
        c.vkAllocateCommandBuffers(device, &allocate_info, &command_buffer),
        "failed to allocate command buffer",
    );
    return command_buffer;
}

const SynchronisationObjects = struct {
    image_available_semaphore: c.VkSemaphore,
    render_finished_semaphore: c.VkSemaphore,
    in_flight_fence: c.VkFence,
};

fn initSynchronisation(device: c.VkDevice) !SynchronisationObjects {
    const semaphore_create_info = c.VkSemaphoreCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    const fence_create_info = c.VkFenceCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = c.VK_FENCE_CREATE_SIGNALED_BIT,
    };

    var image_available_semaphore: c.VkSemaphore = undefined;
    try wrapVulkanResult(
        c.vkCreateSemaphore(device, &semaphore_create_info, null, &image_available_semaphore),
        "failed to create semaphore",
    );
    errdefer c.vkDestroySemaphore(device, image_available_semaphore, null);

    var render_finished_semaphore: c.VkSemaphore = undefined;
    try wrapVulkanResult(
        c.vkCreateSemaphore(device, &semaphore_create_info, null, &render_finished_semaphore),
        "failed to create semaphore",
    );
    errdefer c.vkDestroySemaphore(device, image_available_semaphore, null);

    var in_flight_fence: c.VkFence = undefined;
    try wrapVulkanResult(
        c.vkCreateFence(device, &fence_create_info, null, &in_flight_fence),
        "failed to create fence",
    );

    return .{
        .image_available_semaphore = image_available_semaphore,
        .render_finished_semaphore = render_finished_semaphore,
        .in_flight_fence = in_flight_fence,
    };
}

fn recordCommandBuffer(self: *Self, framebuffer: c.VkFramebuffer) !void {
    const begin_info = c.VkCommandBufferBeginInfo{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0,
    };
    try wrapVulkanResult(
        c.vkBeginCommandBuffer(self.command_buffer, &begin_info),
        "failed to beging recording command buffer",
    );

    const render_pass_info = c.VkRenderPassBeginInfo{
        .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = self.render_pass,
        .framebuffer = framebuffer,
        .renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = self.swapchain_extent },
        .clearValueCount = 1,
        .pClearValues = &c.VkClearValue{ .color = .{ .float32 = [_]f32{ 0.0, 0.0, 0.0, 1.0 } } },
    };
    c.vkCmdBeginRenderPass(self.command_buffer, &render_pass_info, c.VK_SUBPASS_CONTENTS_INLINE);

    c.vkCmdBindPipeline(
        self.command_buffer,
        c.VK_PIPELINE_BIND_POINT_GRAPHICS,
        self.graphics_pipeline,
    );

    c.vkCmdDraw(self.command_buffer, 3, 1, 0, 0);

    c.vkCmdEndRenderPass(self.command_buffer);
    try wrapVulkanResult(
        c.vkEndCommandBuffer(self.command_buffer),
        "failed to record command buffer",
    );
}

pub fn drawFrame(self: *Self) !void {
    try wrapVulkanResult(
        c.vkWaitForFences(self.device, 1, &self.in_flight_fence, c.VK_TRUE, std.math.maxInt(u64)),
        "failed to wait for in-flight fence",
    );
    try wrapVulkanResult(
        c.vkResetFences(self.device, 1, &self.in_flight_fence),
        "failed to reset in-flight fence",
    );

    var image_index: u32 = undefined;
    _ = c.vkAcquireNextImageKHR(
        self.device,
        self.swapchain,
        std.math.maxInt(u64),
        self.image_available_semaphore,
        null,
        &image_index,
    );

    try wrapVulkanResult(
        c.vkResetCommandBuffer(self.command_buffer, 0),
        "failed to reset command buffer",
    );
    try self.recordCommandBuffer(self.framebuffers[image_index]);

    const submit_info = c.VkSubmitInfo{
        .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &self.image_available_semaphore,
        .pWaitDstStageMask = &@intCast(c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
        .commandBufferCount = 1,
        .pCommandBuffers = &self.command_buffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &self.render_finished_semaphore,
    };
    try wrapVulkanResult(
        c.vkQueueSubmit(self.graphics_queue, 1, &submit_info, self.in_flight_fence),
        "failed to submit draw command buffer",
    );

    const present_info = c.VkPresentInfoKHR{
        .sType = c.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &self.render_finished_semaphore,
        .swapchainCount = 1,
        .pSwapchains = &self.swapchain,
        .pImageIndices = &image_index,
    };
    _ = c.vkQueuePresentKHR(self.present_queue, &present_info);
}
