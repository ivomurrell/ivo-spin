const std = @import("std");
const builtin = @import("builtin");

const c = @import("c.zig");
const mat = @import("matrix.zig");
const SDL = @import("sdl.zig");

const Allocator = std.mem.Allocator;

const Self = @This();

const frames_in_flight = 2;

allocator: Allocator,
timer: std.time.Timer,
current_frame: u8 = 0,
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
depth_resources: DepthResources,
render_pass: c.VkRenderPass,
descriptor_set_layout: c.VkDescriptorSetLayout,
pipeline_layout: c.VkPipelineLayout,
graphics_pipeline: c.VkPipeline,
framebuffers: []c.VkFramebuffer,
command_pool: c.VkCommandPool,
vertex_buffer: VertexBuffer,
index_buffer: IndexBuffer,
uniform_buffers: [frames_in_flight]UBOBuffer,
descriptor_pool: c.VkDescriptorPool,
descriptor_sets: [frames_in_flight]c.VkDescriptorSet,
command_buffers: [frames_in_flight]c.VkCommandBuffer,
image_available_semaphores: [frames_in_flight]c.VkSemaphore,
render_finished_semaphores: []c.VkSemaphore,
in_flight_fences: [frames_in_flight]c.VkFence,

pub fn init(allocator: Allocator, sdl: *const SDL) !Self {
    const timer = try std.time.Timer.start();

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
        c.VK_IMAGE_ASPECT_COLOR_BIT,
    );
    errdefer {
        for (swapchain_image_views) |image_view| {
            c.vkDestroyImageView(device, image_view, null);
        }
        allocator.free(swapchain_image_views);
    }

    const depth_resources = try initDepthResources(
        device,
        physical_device,
        swapchain.image_extent,
    );
    errdefer {
        c.vkDestroyImageView(device, depth_resources.image_view, null);
        depth_resources.image.deinit();
    }

    const render_pass = try initRenderPass(
        device,
        swapchain.image_format,
        depth_resources.image.format,
    );
    errdefer c.vkDestroyRenderPass(device, render_pass, null);

    const descriptor_set_layout = try initDescriptorSetLayout(device);
    errdefer c.vkDestroyDescriptorSetLayout(device, descriptor_set_layout, null);

    const pipeline_layout = try initPipelineLayout(device, descriptor_set_layout);
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
        depth_resources.image_view,
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

    const vertex_buffer = try initVertexBuffer(
        device,
        physical_device,
        command_pool,
        graphics_queue,
    );
    errdefer vertex_buffer.deinit();
    const index_buffer = try initIndexBuffer(
        device,
        physical_device,
        command_pool,
        graphics_queue,
    );
    errdefer index_buffer.deinit();

    const uniform_buffers = try initUniformBuffers(device, physical_device);
    errdefer for (uniform_buffers) |uniform_buffer| {
        uniform_buffer.deinit();
    };

    const descriptor_pool = try initDescriptorPool(device);
    errdefer c.vkDestroyDescriptorPool(device, descriptor_pool, null);
    const descriptor_sets = try initDescriptorSets(
        device,
        descriptor_pool,
        descriptor_set_layout,
        uniform_buffers,
    );

    const command_buffers = try initCommandBuffers(device, command_pool);

    const synchronisation_objects = try initSynchronisation(allocator, device, swapchain.images.len);
    errdefer {
        for (synchronisation_objects.image_available_semaphores) |image_available_semaphore| {
            c.vkDestroySemaphore(device, image_available_semaphore, null);
        }
        for (synchronisation_objects.render_finished_semaphores) |render_finished_semaphore| {
            c.vkDestroySemaphore(device, render_finished_semaphore, null);
        }
        allocator.free(synchronisation_objects.render_finished_semaphores);
        for (synchronisation_objects.in_flight_fences) |in_flight_fence| {
            c.vkDestroyFence(device, in_flight_fence, null);
        }
    }

    return Self{
        .allocator = allocator,
        .timer = timer,
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
        .depth_resources = depth_resources,
        .render_pass = render_pass,
        .pipeline_layout = pipeline_layout,
        .descriptor_set_layout = descriptor_set_layout,
        .graphics_pipeline = graphics_pipeline,
        .framebuffers = framebuffers,
        .command_pool = command_pool,
        .vertex_buffer = vertex_buffer,
        .index_buffer = index_buffer,
        .uniform_buffers = uniform_buffers,
        .descriptor_pool = descriptor_pool,
        .descriptor_sets = descriptor_sets,
        .command_buffers = command_buffers,
        .image_available_semaphores = synchronisation_objects.image_available_semaphores,
        .render_finished_semaphores = synchronisation_objects.render_finished_semaphores,
        .in_flight_fences = synchronisation_objects.in_flight_fences,
    };
}

pub fn deinit(self: Self) void {
    wrapVulkanResult(
        c.vkDeviceWaitIdle(self.device),
        "couldn't wait for device to be finished",
    ) catch {};

    for (self.image_available_semaphores) |image_available_semaphore| {
        c.vkDestroySemaphore(self.device, image_available_semaphore, null);
    }
    for (self.render_finished_semaphores) |render_finished_semaphore| {
        c.vkDestroySemaphore(self.device, render_finished_semaphore, null);
    }
    self.allocator.free(self.render_finished_semaphores);
    for (self.in_flight_fences) |in_flight_fence| {
        c.vkDestroyFence(self.device, in_flight_fence, null);
    }
    c.vkDestroyDescriptorPool(self.device, self.descriptor_pool, null);
    for (self.uniform_buffers) |uniform_buffer| {
        uniform_buffer.deinit();
    }
    self.index_buffer.deinit();
    self.vertex_buffer.deinit();
    c.vkDestroyImageView(self.device, self.depth_resources.image_view, null);
    self.depth_resources.image.deinit();
    c.vkDestroyCommandPool(self.device, self.command_pool, null);
    for (self.framebuffers) |framebuffer| {
        c.vkDestroyFramebuffer(self.device, framebuffer, null);
    }
    self.allocator.free(self.framebuffers);
    c.vkDestroyPipeline(self.device, self.graphics_pipeline, null);
    c.vkDestroyPipelineLayout(self.device, self.pipeline_layout, null);
    c.vkDestroyDescriptorSetLayout(self.device, self.descriptor_set_layout, null);
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
    format: c.VkFormat,
    aspect_flags: c.VkImageAspectFlags,
) ![]c.VkImageView {
    const image_views = try allocator.alloc(c.VkImageView, images.len);
    errdefer allocator.free(image_views);

    var num_created: usize = 0;
    errdefer for (image_views[0..num_created]) |image_view| {
        c.vkDestroyImageView(device, image_view, null);
    };
    for (images, image_views) |image, *image_view| {
        image_view.* = try initImageView(
            device,
            image,
            format,
            aspect_flags,
        );
        num_created += 1;
    }

    return image_views;
}

fn initImageView(
    device: c.VkDevice,
    image: c.VkImage,
    format: c.VkFormat,
    aspect_flags: c.VkImageAspectFlags,
) !c.VkImageView {
    var create_info = c.VkImageViewCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = c.VkImageSubresourceRange{
            .aspectMask = aspect_flags,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    var image_view: c.VkImageView = undefined;
    try wrapVulkanResult(
        c.vkCreateImageView(device, &create_info, null, &image_view),
        "failed to create an image view",
    );
    return image_view;
}

const DepthResources = struct {
    image: DepthImage,
    image_view: c.VkImageView,
};

fn initDepthResources(
    device: c.VkDevice,
    physical_device: c.VkPhysicalDevice,
    image_extent: c.VkExtent2D,
) !DepthResources {
    const depth_image = try DepthImage.init(
        device,
        physical_device,
        image_extent.width,
        image_extent.height,
    );
    errdefer depth_image.deinit();

    const depth_image_view = try initImageView(
        device,
        depth_image.handle,
        c.VK_FORMAT_D32_SFLOAT_S8_UINT,
        c.VK_IMAGE_ASPECT_DEPTH_BIT,
    );
    errdefer c.vkDestroyImageView(device, depth_image_view, null);

    return .{
        .image = depth_image,
        .image_view = depth_image_view,
    };
}

fn initRenderPass(
    device: c.VkDevice,
    image_format: c.VkFormat,
    depth_format: c.VkFormat,
) !c.VkRenderPass {
    const colour_attachment = c.VkAttachmentDescription{
        .format = image_format,
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

    const depth_attachment = c.VkAttachmentDescription{
        .format = depth_format,
        .samples = c.VK_SAMPLE_COUNT_1_BIT,
        .loadOp = c.VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp = c.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = c.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    const depth_attachment_ref = c.VkAttachmentReference{
        .attachment = 1,
        .layout = c.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    const subpass = c.VkSubpassDescription{
        .pipelineBindPoint = c.VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colour_attachment_ref,
        .pDepthStencilAttachment = &depth_attachment_ref,
    };
    const subpass_dependency = c.VkSubpassDependency{
        .srcSubpass = c.VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | c.VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        .srcAccessMask = c.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dstStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | c.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        .dstAccessMask = c.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | c.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    };

    const attachments = [_]c.VkAttachmentDescription{
        colour_attachment,
        depth_attachment,
    };
    const create_info = c.VkRenderPassCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = attachments.len,
        .pAttachments = &attachments,
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

fn initPipelineLayout(
    device: c.VkDevice,
    descriptor_set_layout: c.VkDescriptorSetLayout,
) !c.VkPipelineLayout {
    const create_info = c.VkPipelineLayoutCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
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

fn initDescriptorSetLayout(device: c.VkDevice) !c.VkDescriptorSetLayout {
    const ubo_layout_binding = c.VkDescriptorSetLayoutBinding{
        .binding = 0,
        .descriptorType = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = c.VK_SHADER_STAGE_VERTEX_BIT,
    };

    const create_info = c.VkDescriptorSetLayoutCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &ubo_layout_binding,
    };
    var descriptor_set_layout: c.VkDescriptorSetLayout = undefined;
    try wrapVulkanResult(
        c.vkCreateDescriptorSetLayout(device, &create_info, null, &descriptor_set_layout),
        "failed to create descriptor set layout",
    );

    return descriptor_set_layout;
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

    const vertex_binding_description = Vertex.bindingDescription();
    const vertex_attribute_descriptions = Vertex.attributeDescriptions();
    const vertex_input_info = c.VkPipelineVertexInputStateCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertex_binding_description,
        .vertexAttributeDescriptionCount = vertex_attribute_descriptions.len,
        .pVertexAttributeDescriptions = &vertex_attribute_descriptions,
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
    const depth_stencil_info = c.VkPipelineDepthStencilStateCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = c.VK_TRUE,
        .depthWriteEnable = c.VK_TRUE,
        .depthCompareOp = c.VK_COMPARE_OP_LESS,
        .depthBoundsTestEnable = c.VK_FALSE,
        .stencilTestEnable = c.VK_FALSE,
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
        .pDepthStencilState = &depth_stencil_info,
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
    depth_image_view: c.VkImageView,
    render_pass: c.VkRenderPass,
) ![]c.VkFramebuffer {
    const framebuffers = try allocator.alloc(c.VkFramebuffer, image_views.len);

    var num_created: usize = 0;
    errdefer for (framebuffers[0..num_created]) |framebuffer| {
        c.vkDestroyFramebuffer(device, framebuffer, null);
    };
    for (image_views, framebuffers) |image_view, *framebuffer| {
        const attachments = [_]c.VkImageView{ image_view, depth_image_view };
        const create_info = c.VkFramebufferCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = render_pass,
            .attachmentCount = attachments.len,
            .pAttachments = &attachments,
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

fn initVertexBuffer(
    device: c.VkDevice,
    physical_device: c.VkPhysicalDevice,
    command_pool: c.VkCommandPool,
    graphics_queue: c.VkQueue,
) !VertexBuffer {
    const vertices = [_]Vertex{ .{
        .position = [_]f32{ -0.5, -0.5, 0.5 },
        .colour = [_]f32{ 1.0, 0.0, 0.0 },
    }, .{
        .position = [_]f32{ 0.5, -0.5, 0.5 },
        .colour = [_]f32{ 0.0, 1.0, 0.0 },
    }, .{
        .position = [_]f32{ 0.5, 0.5, 0.5 },
        .colour = [_]f32{ 0.0, 0.0, 1.0 },
    }, .{
        .position = [_]f32{ -0.5, 0.5, 0.5 },
        .colour = [_]f32{ 1.0, 1.0, 1.0 },
    }, .{
        .position = [_]f32{ -0.5, -0.5, -0.5 },
        .colour = [_]f32{ 1.0, 1.0, 0.0 },
    }, .{
        .position = [_]f32{ 0.5, -0.5, -0.5 },
        .colour = [_]f32{ 0.0, 1.0, 1.0 },
    }, .{
        .position = [_]f32{ 0.5, 0.5, -0.5 },
        .colour = [_]f32{ 1.0, 0.0, 1.0 },
    }, .{
        .position = [_]f32{ -0.5, 0.5, -0.5 },
        .colour = [_]f32{ 0.5, 0.5, 1.0 },
    } };

    return VertexBuffer.init(&vertices, device, physical_device, command_pool, graphics_queue);
}

fn initIndexBuffer(
    device: c.VkDevice,
    physical_device: c.VkPhysicalDevice,
    command_pool: c.VkCommandPool,
    graphics_queue: c.VkQueue,
) !IndexBuffer {
    const front_face = [_]u16{ 0, 1, 2, 2, 3, 0 };
    const right_face = [_]u16{ 1, 5, 6, 6, 2, 1 };
    const top_face = [_]u16{ 4, 5, 1, 1, 0, 4 };
    const left_face = [_]u16{ 4, 0, 3, 3, 7, 4 };
    const bottom_face = [_]u16{ 3, 2, 6, 6, 7, 3 };
    const back_face = [_]u16{ 5, 4, 7, 7, 6, 5 };
    const indices = front_face ++ right_face ++ top_face ++ left_face ++ bottom_face ++ back_face;

    return IndexBuffer.init(&indices, device, physical_device, command_pool, graphics_queue);
}

fn initUniformBuffers(
    device: c.VkDevice,
    physical_device: c.VkPhysicalDevice,
) ![frames_in_flight]UBOBuffer {
    var uniform_buffers = [_]UBOBuffer{undefined} ** frames_in_flight;
    var num_created: usize = 0;
    errdefer for (uniform_buffers[0..num_created]) |uniform_buffer| {
        uniform_buffer.deinit();
    };
    for (&uniform_buffers) |*uniform_buffer| {
        uniform_buffer.* = try UBOBuffer.init(device, physical_device);
        num_created += 1;
    }
    return uniform_buffers;
}

fn initDescriptorPool(device: c.VkDevice) !c.VkDescriptorPool {
    const pool_size = c.VkDescriptorPoolSize{
        .type = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = frames_in_flight,
    };
    const create_info = c.VkDescriptorPoolCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
        .maxSets = frames_in_flight,
    };
    var descriptor_pool: c.VkDescriptorPool = undefined;
    try wrapVulkanResult(
        c.vkCreateDescriptorPool(device, &create_info, null, &descriptor_pool),
        "Failed to create descriptor pool",
    );
    return descriptor_pool;
}

fn initDescriptorSets(
    device: c.VkDevice,
    descriptor_pool: c.VkDescriptorPool,
    descriptor_set_layout: c.VkDescriptorSetLayout,
    uniform_buffers: [frames_in_flight]UBOBuffer,
) ![frames_in_flight]c.VkDescriptorSet {
    const allocate_info = c.VkDescriptorSetAllocateInfo{
        .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = frames_in_flight,
        .pSetLayouts = &[_]c.VkDescriptorSetLayout{descriptor_set_layout} ** frames_in_flight,
    };
    var descriptor_sets = [_]c.VkDescriptorSet{undefined} ** frames_in_flight;
    try wrapVulkanResult(
        c.vkAllocateDescriptorSets(device, &allocate_info, &descriptor_sets),
        "failed to allocate descriptor sets",
    );

    for (descriptor_sets, uniform_buffers) |descriptor_set, uniform_buffer| {
        const buffer_info = c.VkDescriptorBufferInfo{
            .buffer = uniform_buffer.handle.handle,
            .offset = 0,
            .range = c.VK_WHOLE_SIZE,
        };
        const descriptor_write = c.VkWriteDescriptorSet{
            .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .pBufferInfo = &buffer_info,
        };
        c.vkUpdateDescriptorSets(device, 1, &descriptor_write, 0, null);
    }

    return descriptor_sets;
}

fn initCommandBuffers(
    device: c.VkDevice,
    command_pool: c.VkCommandPool,
) ![frames_in_flight]c.VkCommandBuffer {
    const allocate_info = c.VkCommandBufferAllocateInfo{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = frames_in_flight,
    };
    var command_buffers = [_]c.VkCommandBuffer{undefined} ** frames_in_flight;
    try wrapVulkanResult(
        c.vkAllocateCommandBuffers(device, &allocate_info, &command_buffers),
        "failed to allocate command buffer",
    );
    return command_buffers;
}

const SynchronisationObjects = struct {
    image_available_semaphores: [frames_in_flight]c.VkSemaphore,
    render_finished_semaphores: []c.VkSemaphore,
    in_flight_fences: [frames_in_flight]c.VkFence,
};

fn initSynchronisation(
    allocator: Allocator,
    device: c.VkDevice,
    swapchain_image_count: usize,
) !SynchronisationObjects {
    const semaphore_create_info = c.VkSemaphoreCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    const fence_create_info = c.VkFenceCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = c.VK_FENCE_CREATE_SIGNALED_BIT,
    };

    var ias_created: usize = 0;
    var image_available_semaphores = [_]c.VkSemaphore{undefined} ** frames_in_flight;
    errdefer {
        for (image_available_semaphores[0..ias_created]) |image_available_semaphore| {
            c.vkDestroySemaphore(device, image_available_semaphore, null);
        }
    }
    for (&image_available_semaphores) |*image_available_semaphore| {
        try wrapVulkanResult(
            c.vkCreateSemaphore(device, &semaphore_create_info, null, image_available_semaphore),
            "failed to create semaphore",
        );
        ias_created += 1;
    }

    var rfs_created: usize = 0;
    var render_finished_semaphores = try allocator.alloc(c.VkSemaphore, swapchain_image_count);
    errdefer {
        for (render_finished_semaphores[0..rfs_created]) |render_finished_semaphore| {
            c.vkDestroySemaphore(device, render_finished_semaphore, null);
        }
        allocator.free(render_finished_semaphores);
    }
    for (render_finished_semaphores) |*render_finished_semaphore| {
        try wrapVulkanResult(
            c.vkCreateSemaphore(device, &semaphore_create_info, null, render_finished_semaphore),
            "failed to create semaphore",
        );
        rfs_created += 1;
    }

    var iff_created: usize = 0;
    var in_flight_fences = [_]c.VkFence{undefined} ** frames_in_flight;
    errdefer {
        for (in_flight_fences[0..iff_created]) |in_flight_fence| {
            c.vkDestroyFence(device, in_flight_fence, null);
        }
    }
    for (&in_flight_fences) |*in_flight_fence| {
        try wrapVulkanResult(
            c.vkCreateFence(device, &fence_create_info, null, in_flight_fence),
            "failed to create fence",
        );
        iff_created += 1;
    }

    return .{
        .image_available_semaphores = image_available_semaphores,
        .render_finished_semaphores = render_finished_semaphores,
        .in_flight_fences = in_flight_fences,
    };
}

fn recordCommandBuffer(self: *Self, framebuffer: c.VkFramebuffer) !void {
    const command_buffer = self.command_buffers[self.current_frame];

    const begin_info = c.VkCommandBufferBeginInfo{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0,
    };
    try wrapVulkanResult(
        c.vkBeginCommandBuffer(command_buffer, &begin_info),
        "failed to beging recording command buffer",
    );

    const clear_values = [_]c.VkClearValue{
        .{ .color = .{ .float32 = [_]f32{ 0.0, 0.0, 0.0, 1.0 } } },
        .{ .depthStencil = .{ .depth = 1.0, .stencil = 0.0 } },
    };

    const render_pass_info = c.VkRenderPassBeginInfo{
        .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = self.render_pass,
        .framebuffer = framebuffer,
        .renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = self.swapchain_extent },
        .clearValueCount = clear_values.len,
        .pClearValues = &clear_values,
    };
    c.vkCmdBeginRenderPass(command_buffer, &render_pass_info, c.VK_SUBPASS_CONTENTS_INLINE);

    c.vkCmdBindPipeline(
        command_buffer,
        c.VK_PIPELINE_BIND_POINT_GRAPHICS,
        self.graphics_pipeline,
    );
    c.vkCmdBindVertexBuffers(
        command_buffer,
        0,
        1,
        &self.vertex_buffer.handle.handle,
        &@as(u64, 0),
    );
    c.vkCmdBindIndexBuffer(
        command_buffer,
        self.index_buffer.handle.handle,
        0,
        c.VK_INDEX_TYPE_UINT16,
    );
    c.vkCmdBindDescriptorSets(
        command_buffer,
        c.VK_PIPELINE_BIND_POINT_GRAPHICS,
        self.pipeline_layout,
        0,
        1,
        &self.descriptor_sets[self.current_frame],
        0,
        null,
    );

    c.vkCmdDrawIndexed(command_buffer, @intCast(self.index_buffer.data.len), 1, 0, 0, 0);

    c.vkCmdEndRenderPass(command_buffer);
    try wrapVulkanResult(c.vkEndCommandBuffer(command_buffer), "failed to record command buffer");
}

pub fn drawFrame(self: *Self) !void {
    try wrapVulkanResult(
        c.vkWaitForFences(
            self.device,
            1,
            &self.in_flight_fences[self.current_frame],
            c.VK_TRUE,
            std.math.maxInt(u64),
        ),
        "failed to wait for in-flight fence",
    );
    try wrapVulkanResult(
        c.vkResetFences(self.device, 1, &self.in_flight_fences[self.current_frame]),
        "failed to reset in-flight fence",
    );

    var image_index: u32 = undefined;
    _ = c.vkAcquireNextImageKHR(
        self.device,
        self.swapchain,
        std.math.maxInt(u64),
        self.image_available_semaphores[self.current_frame],
        null,
        &image_index,
    );

    self.updateUniformBuffer();

    try wrapVulkanResult(
        c.vkResetCommandBuffer(self.command_buffers[self.current_frame], 0),
        "failed to reset command buffer",
    );
    try self.recordCommandBuffer(self.framebuffers[image_index]);

    const submit_info = c.VkSubmitInfo{
        .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &self.image_available_semaphores[self.current_frame],
        .pWaitDstStageMask = &@intCast(c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
        .commandBufferCount = 1,
        .pCommandBuffers = &self.command_buffers[self.current_frame],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &self.render_finished_semaphores[image_index],
    };
    try wrapVulkanResult(
        c.vkQueueSubmit(
            self.graphics_queue,
            1,
            &submit_info,
            self.in_flight_fences[self.current_frame],
        ),
        "failed to submit draw command buffer",
    );

    const present_info = c.VkPresentInfoKHR{
        .sType = c.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &self.render_finished_semaphores[image_index],
        .swapchainCount = 1,
        .pSwapchains = &self.swapchain,
        .pImageIndices = &image_index,
    };
    _ = c.vkQueuePresentKHR(self.present_queue, &present_info);

    self.current_frame = (self.current_frame + 1) % frames_in_flight;
}

pub const Vertex = struct {
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
};

pub const Buffer = struct {
    device: c.VkDevice,
    handle: c.VkBuffer,
    memory: c.VkDeviceMemory,

    pub fn init(
        device: c.VkDevice,
        physical_device: c.VkPhysicalDevice,
        buffer_size: u64,
        usage_flags: u32,
        required_prop_flags: u32,
    ) !Buffer {
        const buffer_create_info = c.VkBufferCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = buffer_size,
            .usage = usage_flags,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        };
        var buffer: c.VkBuffer = undefined;
        try wrapVulkanResult(
            c.vkCreateBuffer(device, &buffer_create_info, null, &buffer),
            "failed to create vertex buffer",
        );
        errdefer c.vkDestroyBuffer(device, buffer, null);

        var mem_reqs: c.VkMemoryRequirements = undefined;
        c.vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);
        const memory_type_index =
            findMemoryTypeIndex(
                mem_reqs.memoryTypeBits,
                required_prop_flags,
                physical_device,
            ) orelse return error.VulkanInitFailed;

        const allocate_info = c.VkMemoryAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = mem_reqs.size,
            .memoryTypeIndex = memory_type_index,
        };
        var buffer_memory: c.VkDeviceMemory = undefined;
        try wrapVulkanResult(
            c.vkAllocateMemory(device, &allocate_info, null, &buffer_memory),
            "failed to allocate vertex buffer memory",
        );
        errdefer c.vkFreeMemory(device, buffer_memory, null);

        try wrapVulkanResult(
            c.vkBindBufferMemory(device, buffer, buffer_memory, 0),
            "failed to bind vertx buffer memory",
        );

        return Buffer{
            .device = device,
            .handle = buffer,
            .memory = buffer_memory,
        };
    }

    pub fn deinit(self: Buffer) void {
        c.vkDestroyBuffer(self.device, self.handle, null);
        c.vkFreeMemory(self.device, self.memory, null);
    }

    fn findMemoryTypeIndex(
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
};

fn StagedBuffer(comptime T: type, buffer_type: u32) type {
    return struct {
        const SelfBuffer = @This();
        data: []const T,
        handle: Buffer,

        pub fn init(
            data: []const T,
            device: c.VkDevice,
            physical_device: c.VkPhysicalDevice,
            command_pool: c.VkCommandPool,
            graphics_queue: c.VkQueue,
        ) !SelfBuffer {
            const buffer_size = @sizeOf(T) * data.len;

            const staging_buffer = try Buffer.init(
                device,
                physical_device,
                buffer_size,
                c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            );
            defer staging_buffer.deinit();

            var mapped_data: [*]T = undefined;
            try wrapVulkanResult(
                c.vkMapMemory(
                    device,
                    staging_buffer.memory,
                    0,
                    buffer_size,
                    0,
                    @ptrCast(&mapped_data),
                ),
                "failed to map vertex buffer memory",
            );
            @memcpy(mapped_data, data);
            c.vkUnmapMemory(device, staging_buffer.memory);

            const data_buffer = try Buffer.init(
                device,
                physical_device,
                buffer_size,
                c.VK_IMAGE_USAGE_TRANSFER_DST_BIT | buffer_type,
                c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            );
            errdefer data_buffer.deinit();

            const allocate_info = c.VkCommandBufferAllocateInfo{
                .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandPool = command_pool,
                .commandBufferCount = 1,
            };

            var command_buffer: c.VkCommandBuffer = undefined;
            try wrapVulkanResult(
                c.vkAllocateCommandBuffers(device, &allocate_info, &command_buffer),
                "failed to allocate copying command buffer",
            );
            defer c.vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);

            const begin_info = c.VkCommandBufferBeginInfo{
                .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };
            try wrapVulkanResult(
                c.vkBeginCommandBuffer(command_buffer, &begin_info),
                "failed to begin recording copying command buffer",
            );

            const copy_region = c.VkBufferCopy{
                .size = buffer_size,
            };
            c.vkCmdCopyBuffer(
                command_buffer,
                staging_buffer.handle,
                data_buffer.handle,
                1,
                &copy_region,
            );

            try wrapVulkanResult(
                c.vkEndCommandBuffer(command_buffer),
                "failed to end recording copying command buffer",
            );

            const submit_info = c.VkSubmitInfo{
                .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1,
                .pCommandBuffers = &command_buffer,
            };
            try wrapVulkanResult(
                c.vkQueueSubmit(graphics_queue, 1, &submit_info, null),
                "failed to submit copying command to queue",
            );
            try wrapVulkanResult(
                c.vkQueueWaitIdle(graphics_queue),
                "failed to wait for queue to complete",
            );

            return SelfBuffer{
                .data = data,
                .handle = data_buffer,
            };
        }

        pub fn deinit(self: SelfBuffer) void {
            self.handle.deinit();
        }
    };
}

pub const VertexBuffer = StagedBuffer(Vertex, c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
pub const IndexBuffer = StagedBuffer(u16, c.VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

fn UniformBuffer(comptime T: type) type {
    return struct {
        const SelfBuffer = @This();
        handle: Buffer,
        mapped_memory: *T,

        pub fn init(device: c.VkDevice, physical_device: c.VkPhysicalDevice) !SelfBuffer {
            const buffer_size = @sizeOf(T);
            const buffer = try Buffer.init(
                device,
                physical_device,
                buffer_size,
                c.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            );
            errdefer buffer.deinit();

            var mapped_memory: *T = undefined;
            try wrapVulkanResult(
                c.vkMapMemory(device, buffer.memory, 0, buffer_size, 0, @ptrCast(&mapped_memory)),
                "failed to map uniform buffer memory",
            );

            return SelfBuffer{
                .handle = buffer,
                .mapped_memory = mapped_memory,
            };
        }

        pub fn deinit(self: SelfBuffer) void {
            self.handle.deinit();
        }
    };
}
pub const UniformBufferObject = extern struct {
    model: mat.Mat4,
    view: mat.Mat4,
    projection: mat.Mat4,
};
pub const UBOBuffer = UniformBuffer(UniformBufferObject);

fn updateUniformBuffer(self: *Self) void {
    const time = self.timer.read();
    const time_s = @as(f32, @floatFromInt(time)) / 1_000_000_000;
    const angle = std.math.degreesToRadians(time_s * 20);
    const aspect_ratio = @as(f32, @floatFromInt(self.swapchain_extent.width)) /
        @as(f32, @floatFromInt(self.swapchain_extent.height));

    self.uniform_buffers[self.current_frame].mapped_memory.* = UniformBufferObject{
        .model = mat.Mat4.rotate(angle, angle, angle),
        .view = mat.Mat4.lookAt(.{ 0, 0, -2.5 }, .{ 0, 0, 0 }, .{ 0, -1, 0 }),
        .projection = mat.Mat4.perspective(std.math.degreesToRadians(45), aspect_ratio, 0.1, 10),
    };
}

fn Image(
    format: c.VkFormat,
    tiling: c.VkImageTiling,
    usage: c.VkImageUsageFlags,
    mem_properties: c.VkMemoryPropertyFlags,
) type {
    return struct {
        const SelfImage = @This();
        device: c.VkDevice,
        handle: c.VkImage,
        memory: c.VkDeviceMemory,
        format: c.VkFormat,

        pub fn init(
            device: c.VkDevice,
            physical_device: c.VkPhysicalDevice,
            width: u32,
            height: u32,
        ) !SelfImage {
            const image_create_info = c.VkImageCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType = c.VK_IMAGE_TYPE_2D,
                .format = format,
                .extent = c.VkExtent3D{
                    .width = width,
                    .height = height,
                    .depth = 1,
                },
                .mipLevels = 1,
                .arrayLayers = 1,
                .tiling = tiling,
                .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
                .usage = usage,
                .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
                .samples = c.VK_SAMPLE_COUNT_1_BIT,
            };
            var image: c.VkImage = undefined;
            try wrapVulkanResult(
                c.vkCreateImage(device, &image_create_info, null, &image),
                "failed to create image",
            );
            errdefer c.vkDestroyImage(device, image, null);

            var mem_reqs: c.VkMemoryRequirements = undefined;
            c.vkGetImageMemoryRequirements(device, image, &mem_reqs);
            const memory_type_index = Buffer.findMemoryTypeIndex(
                mem_reqs.memoryTypeBits,
                mem_properties,
                physical_device,
            ) orelse return error.VulkanInitFailed;
            const allocate_info = c.VkMemoryAllocateInfo{
                .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = mem_reqs.size,
                .memoryTypeIndex = memory_type_index,
            };
            var image_memory: c.VkDeviceMemory = undefined;
            try wrapVulkanResult(
                c.vkAllocateMemory(device, &allocate_info, null, &image_memory),
                "failed to allocate image memory",
            );
            errdefer c.vkFreeMemory(device, image_memory, null);
            try wrapVulkanResult(
                c.vkBindImageMemory(device, image, image_memory, 0),
                "failed to bind image memory",
            );

            return SelfImage{
                .device = device,
                .handle = image,
                .memory = image_memory,
                .format = format,
            };
        }

        pub fn deinit(self: SelfImage) void {
            c.vkDestroyImage(self.device, self.handle, null);
            c.vkFreeMemory(self.device, self.memory, null);
        }
    };
}

pub const DepthImage = Image(
    c.VK_FORMAT_D32_SFLOAT_S8_UINT,
    c.VK_IMAGE_TILING_OPTIMAL,
    c.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
    c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
);
