#include "VulkanAdapter.h"
#include <QVulkanInstance>
#include <QByteArrayList>

VulkanAdapter::VulkanAdapter(QWidget* targetWindow)
    : m_targetWindow(targetWindow)
{
}

void VulkanAdapter::Initialize()
{
    // init vulkan library
    volkInitialize();

    // init vulkan instance
    if (!InitVulkanInstance())
    {
        return;
    }

    // init vulkan device
    if (!InitVulkanDevice())
    {
        return;
    }

    // init vulkan queue
    if (!InitVulkanQueue())
    {
        return;
    }

    // init vma allocator
    if (!InitVmaAllocator())
    {
        return;
    }

    // init vulkan shader
    if (!InitVulkanShader())
    {
        return;
    }

    // init vulkan sync objects
    if (!InitVulkanSyncObjects())
    {
        return;
    }

    if (!InitVulkanCommandPools())
    {
        return;
    }
}

bool VulkanAdapter::Check(VkResult result)
{
    if (result != VK_SUCCESS)
    {
        std::cerr << "Vulkan call returned an error (" << result << ")\n";
        return false;
    }
    return true;
}

bool VulkanAdapter::Check(bool result)
{
    if (!result)
    {
        std::cerr << "Call returned an error\n";
        return false;
    }
    return true;
}

bool VulkanAdapter::InitVulkanInstance()
{
    VkApplicationInfo appInfo{
        .sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Vulkan Renderer",
        .apiVersion       = VK_API_VERSION_1_3};

    QVulkanInstance          probe;
    QByteArrayList           qExts = probe.extensions();
    std::vector<const char*> instExtensions;
    instExtensions.reserve(qExts.size());
    for (const QByteArray& e : qExts)
    {
        instExtensions.push_back(e.constData());
    }

    VkInstanceCreateInfo instCI{
        .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo        = &appInfo,
        .enabledExtensionCount   = instExtensions.size(),
        .ppEnabledExtensionNames = instExtensions.data()};

    if (!Check(vkCreateInstance(&instCI, nullptr, &instance)))
    {
        return false;
    }

    volkLoadInstance(instance);
    return true;
}

bool VulkanAdapter::InitVulkanDevice()
{
    const uint32_t deviceIndex = 0;
    uint32_t       deviceCount = 0;
    if (!Check(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr)))
    {
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    if (!Check(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data())))
    {
        return false;
    }
    physicalDevice = devices[deviceIndex];

    return true;
}

bool VulkanAdapter::InitVulkanQueue()
{
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
    for (size_t i = 0; i < queueFamilies.size(); ++i)
    {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            queueFamilyIndex = i;
        }
    }

    const float             queuePriorities = 1.0f;
    VkDeviceQueueCreateInfo queueCI{
        .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = queueFamilyIndex,
        .queueCount       = 1,
        .pQueuePriorities = &queuePriorities};
    VkPhysicalDeviceFeatures         vk10Features{.samplerAnisotropy = VK_TRUE};
    VkPhysicalDeviceVulkan12Features vk12Features{
        .sType                                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .descriptorIndexing                       = true,
        .descriptorBindingVariableDescriptorCount = true,
        .runtimeDescriptorArray                   = true,
        .bufferDeviceAddress                      = true};
    VkPhysicalDeviceVulkan13Features vk13Features{
        .sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext            = &vk12Features,
        .synchronization2 = true,
        .dynamicRendering = true};

    const std::vector<const char*> deviceExts{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo             deviceCI{
                    .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                    .pNext                   = &vk13Features,
                    .queueCreateInfoCount    = 1,
                    .pQueueCreateInfos       = &queueCI,
                    .enabledExtensionCount   = deviceExts.size(),
                    .ppEnabledExtensionNames = deviceExts.data(),
                    .pEnabledFeatures        = &vk10Features};
    if (!Check(vkCreateDevice(physicalDevice, &deviceCI, nullptr, &device)))
    {
        return false;
    }

    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
    return true;
}

bool VulkanAdapter::InitVmaAllocator()
{
    VmaVulkanFunctions     vkFunctions{.vkGetInstanceProcAddr = vkGetInstanceProcAddr,
                                       .vkGetDeviceProcAddr   = vkGetDeviceProcAddr,
                                       .vkCreateImage         = vkCreateImage};
    VmaAllocatorCreateInfo allocatorCI{
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = physicalDevice,
        .device           = device,
        .pVulkanFunctions = &vkFunctions,
        .instance         = instance};
    if (!Check(vmaCreateAllocator(&allocatorCI, &allocator)))
    {
        return false;
    }
    return true;
}

bool VulkanAdapter::InitVulkanSwapchain()
{
    // surface
#ifdef WIN32
    HWND      hwnd  = (HWND)m_targetWindow->winId();
    HINSTANCE hinst = GetModuleHandle(nullptr);

    VkWin32SurfaceCreateInfoKHR surfaceCI{
        .sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
        .hinstance = hinst,
        .hwnd      = hwnd,
    };

    if (!Check(vkCreateWin32SurfaceKHR(instance, &surfaceCI, nullptr, &surface)))
    {
        return false;
    }
#endif

    VkSurfaceCapabilitiesKHR surfaceCaps{};
    if (!Check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCaps)))
    {
        return false;
    }

    // swapchain
    const VkFormat           imageFormat{VK_FORMAT_B8G8R8_SRGB};
    VkSwapchainCreateInfoKHR swapchainCI{
        .sType           = VK_STRUCTURE_TYPE_IMAGE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface         = surface,
        .minImageCount   = surfaceCaps.minImageCount,
        .imageFormat     = imageFormat,
        .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        .imageExtent{.width  = m_targetWindow->width(),
                     .height = m_targetWindow->height()},
        .imageArrayLayers = 1,
        .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .preTransform     = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode      = VK_PRESENT_MODE_FIFO_KHR};
    if (!Check(vkCreateSwapchainKHR(device, &swapchainCI, nullptr, &swapchain)))
    {
        return false;
    }

    uint32_t imageCount = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());
    for (int i = 0; i < imageCount; ++i)
    {
        VkImageViewCreateInfo viewCI{
            .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image    = swapchainImages[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format   = imageFormat,
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .levelCount = 1,
                .layerCount = 1}};
        if (!Check(vkCreateImageView(device, &viewCI, nullptr, &swapchainImageViews[i])))
        {
            return false;
        }
    }

    // depth attachment
    const VkFormat    depthFormat{VK_FORMAT_D24_UNORM_S8_UINT};
    VkImageCreateInfo depthImageCI{
        .sType     = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format    = depthFormat,
        .extent{.width  = m_targetWindow->width(),
                .height = m_targetWindow->height(),
                .depth  = 1},
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo allocCI{
        .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO};
    vmaCreateImage(allocator, &depthImageCI, &allocCI, &depthImage, &depthImageAllocation, nullptr);
    VkImageViewCreateInfo depthViewCI{
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image    = depthImage,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format   = depthFormat,
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
            .levelCount = 1,
            .layerCount = 1}};
    if (!Check(vkCreateImageView(device, &depthViewCI, nullptr, &depthImageView)))
    {
        return false;
    }
    return true;
}

bool VulkanAdapter::InitVulkanShader()
{
    for (int i = 0; i < maxFramesInFlight; ++i)
    {
        VkBufferCreateInfo bufferCI{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size  = sizeof(ShaderData),
            .usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
        VmaAllocationCreateInfo bufferAllocCI{
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO};
        if (!Check(vmaCreateBuffer(allocator, &bufferCI, &bufferAllocCI, &shaderDataBuffers[i].buffer, &shaderDataBuffers[i].allocation, nullptr)))
        {
            return false;
        }
        vmaMapMemory(allocator, shaderDataBuffers[i].allocation, &shaderDataBuffers[i].mapped);

        VkBufferDeviceAddressInfo bufferDAI{
            .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = shaderDataBuffers[i].buffer};
        shaderDataBuffers[i].deviceAddress = vkGetBufferDeviceAddress(device, &bufferDAI);
    }

    // shader compiler
    slang::createGlobalSession(slangGlobalSession.writeRef());
    std::array<slang::TargetDesc, 1> slangTargets{
        slang::TargetDesc{.format{SLANG_SPIRV},
                          .profile{slangGlobalSession->findProfile("spirv_1_4")}}};
    std::array<slang::CompilerOptionEntry, 1> slangOptions{
        slang::CompilerOptionEntry{slang::CompilerOptionName::EmitSpirvDirectly,
                                   slang::CompilerOptionValue{slang::CompilerOptionValueKind::Int, 1}}};
    slang::SessionDesc slangSessionDesc{
        .targets                  = slangTargets.data(),
        .targetCount              = SlangInt(slangTargets.size()),
        .defaultMatrixLayoutMode  = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR,
        .compilerOptionEntries    = slangOptions.data(),
        .compilerOptionEntryCount = slangOptions.size()};
    Slang::ComPtr<slang::ISession> slangSession;
    slangGlobalSession->createSession(slangSessionDesc, slangSession.writeRef());

    Slang::ComPtr<slang::IModule> slangModule{slangSession->loadModuleFromSource("triangle", "assets/shader.slang", nullptr, nullptr)};
    Slang::ComPtr<ISlangBlob>     spirv;
    slangModule->getTargetCode(0, spirv.writeRef());

    VkShaderModuleCreateInfo shaderModuleCI{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv->getBufferSize(),
        .pCode    = (uint32_t*)spirv->getBufferPointer()};
    vkCreateShaderModule(device, &shaderModuleCI, nullptr, &shaderModule);

    return true;
}

bool VulkanAdapter::InitVulkanSyncObjects()
{
    VkFenceCreateInfo fenceCI{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT};
    VkSemaphoreCreateInfo semaphoreCI{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (int i = 0; i < maxFramesInFlight; ++i)
    {
        if (!Check(vkCreateFence(device, &fenceCI, nullptr, &fences[i])))
        {
            return false;
        }

        if (!Check(vkCreateSemaphore(device, &semaphoreCI, nullptr, &presentSemaphores[i])))
        {
            return false;
        }
    }

    renderSemaphores.resize(swapchainImages.size());
    for (auto& semaphore : renderSemaphores)
    {
        if (!Check(vkCreateSemaphore(device, &semaphoreCI, nullptr, &semaphore)))
        {
            return false;
        }
    }
}

bool VulkanAdapter::InitVulkanCommandPools()
{
    VkCommandPoolCreateInfo commandPoolCI{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queueFamilyIndex};
    if (!Check(vkCreateCommandPool(device, &commandPoolCI, nullptr, &commandPool)))
    {
        return false;
    }

    VkCommandBufferAllocateInfo commandBufferAI{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = commandPool,
        .commandBufferCount = maxFramesInFlight};
    if (!Check(vkAllocateCommandBuffers(device, &commandBufferAI, commandBuffers.data())))
    {
        return false;
    }
    return true;
}

bool VulkanAdapter::InitVulkanPipeline()
{
    // VkPushConstantRange pushConstantRange{
    //     .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
    //     .size       = sizeof(VkDeviceAddress)};
    // VkPipelineLayoutCreateInfo pipelineLayoutCI{
    //     .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    //     .setLayoutCount = 1,
    //     .pSetLayouts    =,
    // };
    return true;
}
