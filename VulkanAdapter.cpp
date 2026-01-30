#include "VulkanAdapter.h"
#include "VulkanRendererWidget.h"

#define VOLK_IMPLEMENTATION
#include <volk/volk.h>

#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>

VulkanAdapter::VulkanAdapter(VulkanRendererWidget* targetWindow)
    : m_targetWindow(targetWindow)
{
}

VulkanAdapter::~VulkanAdapter()
{
    Uninitialize();
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

    // init vulkan swapchain
    if (!InitVulkanSwapchain())
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

    // init vulkan command pools
    if (!InitVulkanCommandPools())
    {
        return;
    }

    // init vulkan descriptor set layout
    if (!InitVulkanDescriptorSetLayout())
    {
        return;
    }

    // init vulkan pipeline
    if (!InitVulkanPipeline())
    {
        return;
    }

    m_ready = true;
}

void VulkanAdapter::Uninitialize()
{
    vkDeviceWaitIdle(device);

    for (auto i = 0; i < maxFramesInFlight; i++)
    {
        vkDestroyFence(device, fences[i], nullptr);
        vkDestroySemaphore(device, presentSemaphores[i], nullptr);
        vkDestroySemaphore(device, renderSemaphores[i], nullptr);
        vmaUnmapMemory(allocator, shaderDataBuffers[i].allocation);
        vmaDestroyBuffer(allocator, shaderDataBuffers[i].buffer, shaderDataBuffers[i].allocation);
    }

    vmaDestroyImage(allocator, depthImage, depthImageAllocation);
    vkDestroyImageView(device, depthImageView, nullptr);

    for (auto i = 0; i < swapchainImageViews.size(); i++)
    {
        vkDestroyImageView(device, swapchainImageViews[i], nullptr);
    }

    for (const BufferDesc& bufferDesc : modelBuffers)
    {
        vmaDestroyBuffer(allocator, bufferDesc.buffer, bufferDesc.allocation);
    }

    // TODO: destroy textures

    vkDestroyDescriptorSetLayout(device, descriptorSetLayoutTex, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyShaderModule(device, shaderModule, nullptr);
    vmaDestroyAllocator(allocator);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
}

bool VulkanAdapter::IsReady()
{
    return m_ready;
}

void VulkanAdapter::Tick(double elapsed)
{
    std::lock_guard<std::mutex> guard(renderLock);

    PollInputEvents(elapsed);

    if (updateSwapchain)
    {
        UpdateSwapchain();
    }

    if (!Check(vkWaitForFences(device, 1, &fences[frameIndex], true, UINT64_MAX)) ||
        !Check(vkResetFences(device, 1, &fences[frameIndex])))
    {
        return;
    }

    if (!CheckSwapchain(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, presentSemaphores[frameIndex], VK_NULL_HANDLE, &imageIndex)))
    {
        return;
    }

    // update shader data
    float aspect          = (float)m_targetWindow->width() / m_targetWindow->height();
    shaderData.projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 3200.0f);
    shaderData.projection[1][1] *= -1;
    shaderData.view = glm::translate(glm::mat4(1), camPos);

    shaderData.model = glm::translate(glm::mat4(1.0f), modelPos) * glm::mat4_cast(glm::quat(modelRotation));

    memcpy(shaderDataBuffers[frameIndex].mapped, &shaderData, sizeof(ShaderData));

    // command buffers
    VkCommandBuffer commandBuffer = commandBuffers[frameIndex];
    if (!Check(vkResetCommandBuffer(commandBuffer, 0)))
    {
        return;
    }
    VkCommandBufferBeginInfo cbBI{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if (!Check(vkBeginCommandBuffer(commandBuffer, &cbBI)))
    {
        return;
    }

    std::array<VkImageMemoryBarrier2, 2> outputBarriers{
        VkImageMemoryBarrier2{
            .sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .image     = swapchainImages[imageIndex],
            .subresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                              .levelCount = 1,
                              .layerCount = 1}},
        VkImageMemoryBarrier2{
            .sType        = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstStageMask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout     = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
            .image         = depthImage,
            .subresourceRange{.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT |
                                            VK_IMAGE_ASPECT_STENCIL_BIT,
                              .levelCount = 1,
                              .layerCount = 1}}};
    VkDependencyInfo barrierDependencyInfo{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 2,
        .pImageMemoryBarriers    = outputBarriers.data()};
    vkCmdPipelineBarrier2(commandBuffer, &barrierDependencyInfo);

    VkRenderingAttachmentInfo colorAttachmentInfo{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = swapchainImageViews[imageIndex],
        .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue{.color{0.0f, 0.0f, 0.0f, 1.0f}}};
    VkRenderingAttachmentInfo depthAttachmentInfo{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = depthImageView,
        .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .clearValue  = {.depthStencil = {1.0f, 0}}};
    VkRenderingInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea{
            .extent{.width  = m_targetWindow->GetWidthPix(),
                    .height = m_targetWindow->GetHeightPix()}},
        .layerCount           = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &colorAttachmentInfo,
        .pDepthAttachment     = &depthAttachmentInfo};
    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    VkViewport viewport{.width    = static_cast<float>(m_targetWindow->GetWidthPix()),
                        .height   = static_cast<float>(m_targetWindow->GetHeightPix()),
                        .minDepth = 0.0f,
                        .maxDepth = 1.0f};
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{
        .extent{.width  = m_targetWindow->GetWidthPix(),
                .height = m_targetWindow->GetHeightPix()}};
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                            0, 1, &descriptorSetTex, 0, nullptr);

    for (const BufferDesc& bufferDesc : modelBuffers)
    {
        VkDeviceSize vOffset = 0;
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &bufferDesc.buffer, &vOffset);
        vkCmdBindIndexBuffer(commandBuffer, bufferDesc.buffer, bufferDesc.offsetOfIndexBuffer, VK_INDEX_TYPE_UINT32);
        vkCmdPushConstants(commandBuffer,
                           pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT,
                           0,
                           sizeof(VkDeviceAddress),
                           &shaderDataBuffers[frameIndex].deviceAddress);
        vkCmdDrawIndexed(commandBuffer, bufferDesc.indexCount, 1, 0, 0, 0);
    }

    vkCmdEndRendering(commandBuffer);

    VkImageMemoryBarrier2 barrierPresent{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = 0,
        .oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .image         = swapchainImages[imageIndex],
        .subresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                          .levelCount = 1,
                          .layerCount = 1}};
    VkDependencyInfo barrierPresentDependencyInfo{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &barrierPresent};
    vkCmdPipelineBarrier2(commandBuffer, &barrierPresentDependencyInfo);
    if (!Check(vkEndCommandBuffer(commandBuffer)))
    {
        return;
    }

    VkPipelineStageFlags waitStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo         submitInfo{
                .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .waitSemaphoreCount   = 1,
                .pWaitSemaphores      = &presentSemaphores[frameIndex],
                .pWaitDstStageMask    = &waitStages,
                .commandBufferCount   = 1,
                .pCommandBuffers      = &commandBuffer,
                .signalSemaphoreCount = 1,
                .pSignalSemaphores    = &renderSemaphores[imageIndex],
    };
    if (!Check(vkQueueSubmit(queue, 1, &submitInfo, fences[frameIndex])))
    {
        return;
    }

    frameIndex = (frameIndex + 1) % maxFramesInFlight;
    VkPresentInfoKHR presentInfo{.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                                 .waitSemaphoreCount = 1,
                                 .pWaitSemaphores    = &renderSemaphores[imageIndex],
                                 .swapchainCount     = 1,
                                 .pSwapchains        = &swapchain,
                                 .pImageIndices      = &imageIndex};

    if (!CheckSwapchain(vkQueuePresentKHR(queue, &presentInfo)))
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

bool VulkanAdapter::CheckSwapchain(VkResult result)
{
    if (result < VK_SUCCESS)
    {
        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            updateSwapchain = true;
            return true;
        }
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

    std::vector<const char*> instExtensions{"VK_KHR_surface"};
#ifdef WIN32
    instExtensions.push_back("VK_KHR_win32_surface");
#endif

#ifdef _DEBUG
    const std::vector<const char*> instLayers = {"VK_LAYER_KHRONOS_validation"};
#endif

    VkInstanceCreateInfo instCI{
        .sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,

#ifdef _DEBUG
        .enabledLayerCount   = (uint32_t)instLayers.size(),
        .ppEnabledLayerNames = instLayers.data(),
#endif

        .enabledExtensionCount   = (uint32_t)instExtensions.size(),
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
    uint32_t deviceCount = 0;
    if (!Check(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr)) ||
        deviceCount == 0)
    {
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    if (!Check(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data())))
    {
        return false;
    }

    for (VkPhysicalDevice device : devices)
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(device, &props);

        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            physicalDevice = device;
            return true;
        }
    }

    physicalDevice = devices[0];
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
                    .enabledExtensionCount   = (uint32_t)deviceExts.size(),
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
    const VkFormat           imageFormat{VK_FORMAT_B8G8R8A8_SRGB};
    VkSwapchainCreateInfoKHR swapchainCI{
        .sType           = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface         = surface,
        .minImageCount   = surfaceCaps.minImageCount,
        .imageFormat     = imageFormat,
        .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        .imageExtent{.width  = m_targetWindow->GetWidthPix(),
                     .height = m_targetWindow->GetHeightPix()},
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
    swapchainImageViews.resize(imageCount);
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
        .extent{.width  = m_targetWindow->GetWidthPix(),
                .height = m_targetWindow->GetHeightPix(),
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

    Slang::ComPtr<slang::IModule> slangModule{slangSession->loadModuleFromSource("triangle", "resources/shaders/shader.slang", nullptr, nullptr)};
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
    return true;
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

bool VulkanAdapter::InitVulkanDescriptorSetLayout()
{
    VkDescriptorBindingFlags                    descVariableFlag{VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT};
    VkDescriptorSetLayoutBindingFlagsCreateInfo descBindingFlagsCI{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount  = 1,
        .pBindingFlags = &descVariableFlag};
    VkDescriptorSetLayoutBinding descLayoutBidingTex{
        .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1024, // max textures
        .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT};
    VkDescriptorSetLayoutCreateInfo descLayoutTexCI{
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext        = &descBindingFlagsCI,
        .bindingCount = 1,
        .pBindings    = &descLayoutBidingTex};
    if (!Check(vkCreateDescriptorSetLayout(device, &descLayoutTexCI, nullptr, &descriptorSetLayoutTex)))
    {
        return false;
    }

    VkDescriptorPoolSize poolSize{
        .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1024};
    VkDescriptorPoolCreateInfo descPoolCI{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = 1,
        .poolSizeCount = 1,
        .pPoolSizes    = &poolSize};
    if (!Check(vkCreateDescriptorPool(device, &descPoolCI, nullptr, &descriptorPool)))
    {
        return false;
    }

    uint32_t                                           variableDescCount{1024};
    VkDescriptorSetVariableDescriptorCountAllocateInfo variableDescCountAI{
        .sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT,
        .descriptorSetCount = 1,
        .pDescriptorCounts  = &variableDescCount};
    VkDescriptorSetAllocateInfo texDescSetAlloc{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext              = &variableDescCountAI,
        .descriptorPool     = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &descriptorSetLayoutTex};
    if (!Check(vkAllocateDescriptorSets(device, &texDescSetAlloc, &descriptorSetTex)))
    {
        return false;
    }

    // VkWriteDescriptorSet writeDescSet{
    //     .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    //     .dstSet          = descriptorSetTex,
    //     .dstBinding      = 0,
    //     .descriptorCount = 0,
    //     .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    //     .pImageInfo      = nullptr};
    // vkUpdateDescriptorSets(device, 1, &writeDescSet, 0, nullptr);

    return true;
}

bool VulkanAdapter::InitVulkanPipeline()
{
    VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .size       = sizeof(VkDeviceAddress)};
    VkPipelineLayoutCreateInfo pipelineLayoutCI{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &descriptorSetLayoutTex,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstantRange};
    if (!Check(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout)))
    {
        return false;
    }

    std::vector<VkPipelineShaderStageCreateInfo> shaderStagesCIs{
        VkPipelineShaderStageCreateInfo{
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_VERTEX_BIT,
            .module = shaderModule,
            .pName  = "main"},
        VkPipelineShaderStageCreateInfo{
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = shaderModule,
            .pName  = "main"}};
    VkVertexInputBindingDescription vertexBinding{
        .binding   = 0,
        .stride    = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
    std::vector<VkVertexInputAttributeDescription> vertexAttributes{
        VkVertexInputAttributeDescription{
            .location = 0,
            .binding  = 0,
            .format   = VK_FORMAT_R32G32B32_SFLOAT},
        VkVertexInputAttributeDescription{
            .location = 1,
            .binding  = 0,
            .format   = VK_FORMAT_R32G32B32_SFLOAT,
            .offset   = offsetof(Vertex, normal)},
        VkVertexInputAttributeDescription{
            .location = 2,
            .binding  = 0,
            .format   = VK_FORMAT_R32G32_SFLOAT,
            .offset   = offsetof(Vertex, uv)}};
    VkPipelineVertexInputStateCreateInfo vertexInputStageCI{
        .sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount   = 1,
        .pVertexBindingDescriptions      = &vertexBinding,
        .vertexAttributeDescriptionCount = (uint32_t)vertexAttributes.size(),
        .pVertexAttributeDescriptions    = vertexAttributes.data()};
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{
        .sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    std::vector<VkDynamicState> dynamicStates{
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicStateCI{
        .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = (uint32_t)dynamicStates.size(),
        .pDynamicStates    = dynamicStates.data()};
    VkPipelineViewportStateCreateInfo viewportStateCI{
        .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount  = 1};
    VkPipelineRasterizationStateCreateInfo rasterizationStateCI{
        .sType     = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo multisampleStateCI{
        .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineDepthStencilStateCreateInfo depthStencilStateCI{
        .sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable  = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL};
    VkPipelineColorBlendAttachmentState blendAttachment{
        .colorWriteMask = 0xF};
    VkPipelineColorBlendStateCreateInfo colorBlendStateCI{
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments    = &blendAttachment};
    const VkFormat                imageFormat{VK_FORMAT_B8G8R8A8_SRGB};
    const VkFormat                depthFormat{VK_FORMAT_D24_UNORM_S8_UINT};
    VkPipelineRenderingCreateInfo renderingCI{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &imageFormat,
        .depthAttachmentFormat   = depthFormat};
    VkGraphicsPipelineCreateInfo graphicsPipelineCI{
        .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext               = &renderingCI,
        .stageCount          = 2,
        .pStages             = shaderStagesCIs.data(),
        .pVertexInputState   = &vertexInputStageCI,
        .pInputAssemblyState = &inputAssemblyStateCI,
        .pViewportState      = &viewportStateCI,
        .pRasterizationState = &rasterizationStateCI,
        .pMultisampleState   = &multisampleStateCI,
        .pDepthStencilState  = &depthStencilStateCI,
        .pColorBlendState    = &colorBlendStateCI,
        .pDynamicState       = &dynamicStateCI,
        .layout              = pipelineLayout};
    if (!Check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphicsPipelineCI, nullptr, &pipeline)))
    {
        return false;
    }
    return true;
}

void VulkanAdapter::PollInputEvents(double elapsed)
{
    while (auto eventOpt = m_targetWindow->PollEvent())
    {
        Event event = eventOpt.value();
        if (event.type == EventType::ResizeEvent)
        {
            updateSwapchain = true;
        }
        else if (event.type == EventType::DropEvent)
        {
            DropEventData data = std::any_cast<DropEventData>(event.data);
            std::thread   loadThread([this, data]() {
                Model model;
                if (m_parser.Parse(data.file, model))
                {
                    LoadModel(model);
                }
            });
            loadThread.detach();
        }
        else if (event.type == EventType::MouseEvent)
        {
            MouseEventData data = std::any_cast<MouseEventData>(event.data);
            if (data.event == MouseEventType::Move)
            {
                if (data.leftBtnPressing)
                {
                    const float sensitivity = 0.5f;
                    modelRotation.x += data.deltaY * elapsed * sensitivity;
                    modelRotation.y += data.deltaX * elapsed * sensitivity;
                }
                else if (data.rightBtnPressing)
                {
                    modelPos += glm::vec3(data.deltaX * elapsed, -data.deltaY * elapsed, 0);
                }
            }
            else if (data.event == MouseEventType::Wheel)
            {
                glm::vec3   pivot   = modelPos;
                glm::vec3   dir     = glm::normalize(camPos - pivot);
                float       dist    = glm::length(camPos - pivot);
                const float minDist = 0.05f;
                const float maxDist = 1e6f;
                float       steps   = data.deltaY / 120.0f;
                const float k       = 0.15f;
                float       newDist = dist * std::pow(1.0f - k, steps);
                newDist             = std::clamp(newDist, minDist, maxDist);
                camPos              = pivot + dir * newDist;
            }
        }
    }
}

bool VulkanAdapter::UpdateSwapchain()
{
    vkDeviceWaitIdle(device);

    // recreate swapchain
    VkSurfaceCapabilitiesKHR surfaceCaps{};
    if (!Check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCaps)))
    {
        return false;
    }
    const VkFormat           imageFormat{VK_FORMAT_B8G8R8A8_SRGB};
    VkSwapchainCreateInfoKHR swapchainCI{
        .sType           = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface         = surface,
        .minImageCount   = surfaceCaps.minImageCount,
        .imageFormat     = imageFormat,
        .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        .imageExtent{.width  = m_targetWindow->GetWidthPix(),
                     .height = m_targetWindow->GetHeightPix()},
        .imageArrayLayers = 1,
        .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .preTransform     = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode      = VK_PRESENT_MODE_FIFO_KHR,
        .oldSwapchain     = swapchain};
    if (!Check(vkCreateSwapchainKHR(device, &swapchainCI, nullptr, &swapchain)))
    {
        return false;
    }

    // recreate swapchain image views
    for (auto i = 0; i < swapchainImageViews.size(); i++)
    {
        vkDestroyImageView(device, swapchainImageViews[i], nullptr);
    }
    uint32_t imageCount = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());
    swapchainImageViews.resize(imageCount);
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

    // recreate depth attachment
    vkDestroySwapchainKHR(device, swapchainCI.oldSwapchain, nullptr);
    vmaDestroyImage(allocator, depthImage, depthImageAllocation);
    vkDestroyImageView(device, depthImageView, nullptr);
    const VkFormat    depthFormat{VK_FORMAT_D24_UNORM_S8_UINT};
    VkImageCreateInfo depthImageCI{
        .sType     = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format    = depthFormat,
        .extent{.width  = m_targetWindow->GetWidthPix(),
                .height = m_targetWindow->GetHeightPix(),
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

    updateSwapchain = false;
    return true;
}

void VulkanAdapter::LoadModel(const Model& model)
{
    std::lock_guard<std::mutex> guard(renderLock);

    vkDeviceWaitIdle(device);

    std::vector<BufferDesc> newModelBuffers;
    for (const Shape& shape : model.shapes)
    {
        VkDeviceSize       vbSize = sizeof(Vertex) * shape.vertices.size();
        VkDeviceSize       ibSize = sizeof(Index) * shape.indices.size();
        VkBufferCreateInfo bufferCI{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size  = vbSize + ibSize,
            .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDEX_BUFFER_BIT};
        VmaAllocationCreateInfo bufferAllocCI{
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO};

        BufferDesc bufferDesc;
        if (Check(vmaCreateBuffer(allocator, &bufferCI, &bufferAllocCI, &bufferDesc.buffer, &bufferDesc.allocation, nullptr)))
        {
            bufferDesc.offsetOfIndexBuffer = vbSize;
            bufferDesc.indexCount          = shape.indices.size();

            void* mapped = nullptr;
            vmaMapMemory(allocator, bufferDesc.allocation, &mapped);
            memcpy(mapped, shape.vertices.data(), vbSize);
            memcpy(((char*)mapped) + vbSize, shape.indices.data(), ibSize);
            vmaUnmapMemory(allocator, bufferDesc.allocation);

            newModelBuffers.emplace_back(std::move(bufferDesc));
        }
    }

    std::swap(modelBuffers, newModelBuffers);

    modelPos = model.aabb.Center();
    camPos   = modelPos + glm::vec3(0, 0, -model.aabb.Length().z * 2.5f);

    for (const BufferDesc& bufferDesc : newModelBuffers)
    {
        vmaDestroyBuffer(allocator, bufferDesc.buffer, bufferDesc.allocation);
    }
}
