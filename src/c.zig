pub usingnamespace @cImport({
    @cInclude("SDL3/SDL.h");
    @cInclude("SDL3/SDL_vulkan.h");
    @cDefine("VK_ENABLE_BETA_EXTENSIONS", "1");
    @cInclude("vulkan/vulkan.h");
    @cInclude("vulkan/vk_enum_string_helper.h");
});
