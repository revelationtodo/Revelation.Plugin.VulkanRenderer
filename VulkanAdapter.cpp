#include "VulkanAdapter.h"
#include "VulkanRendererWidget.h"

#define VOLK_IMPLEMENTATION
#include <volk/volk.h>

#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>

#include <ktx.h>

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
    if (!InitVulkanMeshShader() || !InitVulkanLineShader() || !InitVulkanSkyboxShader())
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
    if (!InitVulkanMeshDescriptorSetLayout() || !InitVulkanSkyboxDescriptorSetLayout())
    {
        return;
    }

    // init vulkan pipeline
    if (!InitVulkanMeshPipeline() || !InitVulkanLinePipeline() || !InitVulkanSkyboxPipeline())
    {
        return;
    }

    GenerateAxisGridBuffer();

    GenerateSkyboxBuffer();
    LoadSkybox("resources/skybox/cloudy01.ktx2");

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
        vmaUnmapMemory(allocator, frameUniformGpuBuffers[i].allocation);
        vmaDestroyBuffer(allocator, frameUniformGpuBuffers[i].buffer, frameUniformGpuBuffers[i].allocation);
    }

    if (nullptr != meshUniformsGpuBuffer.mapped)
    {
        vmaUnmapMemory(allocator, meshUniformsGpuBuffer.allocation);
    }
    if (VK_NULL_HANDLE != meshUniformsGpuBuffer.buffer)
    {
        vmaDestroyBuffer(allocator, meshUniformsGpuBuffer.buffer, meshUniformsGpuBuffer.allocation);
    }

    vkDestroyDescriptorSetLayout(device, descriptorSetLayoutMat, nullptr);
    vkDestroyDescriptorPool(device, descriptorPoolMat, nullptr);

    vmaDestroyImage(allocator, depthImage, depthImageAllocation);
    vkDestroyImageView(device, depthImageView, nullptr);

    for (auto i = 0; i < swapchainImageViews.size(); i++)
    {
        vkDestroyImageView(device, swapchainImageViews[i], nullptr);
    }

    for (const MeshGpuBuffer& bufferDesc : meshBuffers)
    {
        vmaDestroyBuffer(allocator, bufferDesc.buffer, bufferDesc.allocation);
    }

    vmaDestroyBuffer(allocator, skyboxBuffer.buffer, skyboxBuffer.allocation);

    for (const TextureResource& texture : meshTextures)
    {
        vkDestroyImageView(device, texture.view, nullptr);
        vkDestroySampler(device, texture.sampler, nullptr);
        vmaDestroyImage(allocator, texture.image, texture.allocation);
    }

    vkDestroyImageView(device, skyboxTexture.view, nullptr);
    vkDestroySampler(device, skyboxTexture.sampler, nullptr);
    vmaDestroyImage(allocator, skyboxTexture.image, skyboxTexture.allocation);

    vkDestroyDescriptorSetLayout(device, descriptorSetLayoutTex, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayoutSky, nullptr);
    vkDestroyDescriptorPool(device, descriptorPoolTex, nullptr);
    vkDestroyDescriptorPool(device, descriptorPoolSky, nullptr);
    vkDestroyPipelineLayout(device, meshPipelineLayout, nullptr);
    vkDestroyPipeline(device, meshPipeline, nullptr);
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyShaderModule(device, meshShader, nullptr);
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
    FrameUniforms frameUniforms;
    float         aspect     = (float)m_targetWindow->width() / m_targetWindow->height();
    frameUniforms.projection = glm::perspective(glm::radians(45.0f), aspect, 0.01f, 100000.0f);
    frameUniforms.projection[1][1] *= -1;
    frameUniforms.view      = glm::lookAt(camPosition, navigation, glm::vec3(0, 1, 0));
    frameUniforms.cameraPos = glm::vec4(camPosition, 1.0f);
    memcpy(frameUniformGpuBuffers[frameIndex].mapped, &frameUniforms, sizeof(FrameUniforms));

    PushConstant pushConstant;
    pushConstant.frameUniformsAddr = frameUniformGpuBuffers[frameIndex].deviceAddress;

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
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    // --- draw skybox begin ---
    if (skyboxBuffer.buffer != VK_NULL_HANDLE && skyboxPipeline != VK_NULL_HANDLE && descriptorSetSky != VK_NULL_HANDLE)
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipelineLayout, 0, 1, &descriptorSetSky, 0, nullptr);
        vkCmdPushConstants(commandBuffer, skyboxPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstant), &pushConstant);
        VkDeviceSize vOffset = 0;
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &skyboxBuffer.buffer, &vOffset);
        vkCmdBindIndexBuffer(commandBuffer, skyboxBuffer.buffer, skyboxBuffer.offsetOfIndexBuffer, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer, skyboxBuffer.indexCount, 1, 0, 0, 0);
    }
    // --- draw skybox end ---

    // --- draw axis and grid begin ---
    if (linePipeline != VK_NULL_HANDLE && axisGridBuffer.buffer != VK_NULL_HANDLE && axisGridBuffer.indexCount > 0)
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, linePipeline);
        vkCmdPushConstants(commandBuffer, linePipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstant), &pushConstant);
        VkDeviceSize vOffset = 0;
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &axisGridBuffer.buffer, &vOffset);
        vkCmdDraw(commandBuffer, axisGridBuffer.indexCount, 1, 0, 0);
    }
    // --- draw axis and grid end ---

    // --- draw mesh begin ---
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline);
    std::array<VkDescriptorSet, 2> descSets{descriptorSetTex, descriptorSetMat};
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineLayout, 0, (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
    for (int i = 0; i < meshBuffers.size(); ++i)
    {
        pushConstant.meshUniformsIndex = i;

        const MeshGpuBuffer& meshBuffer = meshBuffers[i];
        VkDeviceSize         vOffset    = 0;
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &meshBuffer.buffer, &vOffset);
        vkCmdBindIndexBuffer(commandBuffer, meshBuffer.buffer, meshBuffer.offsetOfIndexBuffer, VK_INDEX_TYPE_UINT32);
        vkCmdPushConstants(commandBuffer, meshPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstant), &pushConstant);
        vkCmdDrawIndexed(commandBuffer, meshBuffer.indexCount, 1, 0, 0, 0);
    }
    // --- draw mesh end ---

    vkCmdEndRendering(commandBuffer);

    VkImageMemoryBarrier2 barrierPresent{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = 0,
        .oldLayout     = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
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
    for (uint32_t i = 0; i < queueFamilies.size(); ++i)
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
    for (uint32_t i = 0; i < imageCount; ++i)
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

bool VulkanAdapter::InitVulkanMeshShader()
{
    for (int i = 0; i < maxFramesInFlight; ++i)
    {
        VkBufferCreateInfo bufferCI{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size  = sizeof(FrameUniforms),
            .usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
        VmaAllocationCreateInfo bufferAllocCI{
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO};
        if (!Check(vmaCreateBuffer(allocator, &bufferCI, &bufferAllocCI, &frameUniformGpuBuffers[i].buffer, &frameUniformGpuBuffers[i].allocation, nullptr)))
        {
            return false;
        }
        vmaMapMemory(allocator, frameUniformGpuBuffers[i].allocation, &frameUniformGpuBuffers[i].mapped);

        VkBufferDeviceAddressInfo bufferDAI{
            .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = frameUniformGpuBuffers[i].buffer};
        frameUniformGpuBuffers[i].deviceAddress = vkGetBufferDeviceAddress(device, &bufferDAI);
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
        .compilerOptionEntryCount = (uint32_t)slangOptions.size()};
    Slang::ComPtr<slang::ISession> slangSession;
    slangGlobalSession->createSession(slangSessionDesc, slangSession.writeRef());

    Slang::ComPtr<slang::IModule> slangModule{slangSession->loadModuleFromSource("triangle", "resources/shader/mesh_shader.slang", nullptr, nullptr)};
    if (!slangModule)
    {
        std::cerr << "Failed to load mesh_shader.slang\n";
        return false;
    }

    Slang::ComPtr<ISlangBlob> spirv;
    slangModule->getTargetCode(0, spirv.writeRef());
    if (!spirv)
    {
        std::cerr << "Failed to compile mesh shader to SPIR-V\n";
        return false;
    }

    VkShaderModuleCreateInfo shaderModuleCI{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv->getBufferSize(),
        .pCode    = (uint32_t*)spirv->getBufferPointer()};
    vkCreateShaderModule(device, &shaderModuleCI, nullptr, &meshShader);

    return true;
}

bool VulkanAdapter::InitVulkanLineShader()
{
    // shader compiler
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
        .compilerOptionEntryCount = (uint32_t)slangOptions.size()};
    Slang::ComPtr<slang::ISession> slangSession;
    slangGlobalSession->createSession(slangSessionDesc, slangSession.writeRef());

    Slang::ComPtr<slang::IModule> slangModule{slangSession->loadModuleFromSource("line", "resources/shader/line_shader.slang", nullptr, nullptr)};
    if (!slangModule)
    {
        std::cerr << "Failed to load line_shader.slang\n";
        return false;
    }

    Slang::ComPtr<ISlangBlob> spirv;
    slangModule->getTargetCode(0, spirv.writeRef());
    if (!spirv)
    {
        std::cerr << "Failed to compile line shader to SPIR-V\n";
        return false;
    }

    VkShaderModuleCreateInfo shaderModuleCI{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv->getBufferSize(),
        .pCode    = (uint32_t*)spirv->getBufferPointer()};
    if (!Check(vkCreateShaderModule(device, &shaderModuleCI, nullptr, &lineShader)))
    {
        return false;
    }

    return true;
}

bool VulkanAdapter::InitVulkanSkyboxShader()
{
    // shader compiler
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
        .compilerOptionEntryCount = (uint32_t)slangOptions.size()};
    Slang::ComPtr<slang::ISession> slangSession;
    slangGlobalSession->createSession(slangSessionDesc, slangSession.writeRef());

    Slang::ComPtr<slang::IModule> slangModule{slangSession->loadModuleFromSource("triangle", "resources/shader/skybox_shader.slang", nullptr, nullptr)};
    if (!slangModule)
    {
        std::cerr << "Failed to load skybox_shader.slang\n";
        return false;
    }

    Slang::ComPtr<ISlangBlob> spirv;
    slangModule->getTargetCode(0, spirv.writeRef());
    if (!spirv)
    {
        std::cerr << "Failed to compile skybox shader to SPIR-V\n";
        return false;
    }

    VkShaderModuleCreateInfo shaderModuleCI{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv->getBufferSize(),
        .pCode    = (uint32_t*)spirv->getBufferPointer()};
    vkCreateShaderModule(device, &shaderModuleCI, nullptr, &skyboxShader);

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

bool VulkanAdapter::InitVulkanMeshDescriptorSetLayout()
{
    // texture
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

    VkDescriptorPoolSize descPoolSizeTex{
        .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1024};
    VkDescriptorPoolCreateInfo descPoolTexCI{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = 1,
        .poolSizeCount = 1,
        .pPoolSizes    = &descPoolSizeTex};
    if (!Check(vkCreateDescriptorPool(device, &descPoolTexCI, nullptr, &descriptorPoolTex)))
    {
        return false;
    }

    uint32_t                                           variableDescCount = 1024;
    VkDescriptorSetVariableDescriptorCountAllocateInfo variableDescCountAI{
        .sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT,
        .descriptorSetCount = 1,
        .pDescriptorCounts  = &variableDescCount};
    VkDescriptorSetAllocateInfo descTextAI{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext              = &variableDescCountAI,
        .descriptorPool     = descriptorPoolTex,
        .descriptorSetCount = 1,
        .pSetLayouts        = &descriptorSetLayoutTex};
    if (!Check(vkAllocateDescriptorSets(device, &descTextAI, &descriptorSetTex)))
    {
        return false;
    }

    // material ssbo
    VkDescriptorSetLayoutBinding descLayoutBidingMat{
        .binding         = 0,
        .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT};
    VkDescriptorSetLayoutCreateInfo descLayoutMatCI{
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings    = &descLayoutBidingMat};
    if (!Check(vkCreateDescriptorSetLayout(device, &descLayoutMatCI, nullptr, &descriptorSetLayoutMat)))
    {
        return false;
    }

    VkDescriptorPoolSize descPoolSizeMat{
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1};
    VkDescriptorPoolCreateInfo descPoolMatCI{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = 1,
        .poolSizeCount = 1,
        .pPoolSizes    = &descPoolSizeMat};
    if (!Check(vkCreateDescriptorPool(device, &descPoolMatCI, nullptr, &descriptorPoolMat)))
    {
        return false;
    }

    VkDescriptorSetAllocateInfo descMatAI{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = descriptorPoolMat,
        .descriptorSetCount = 1,
        .pSetLayouts        = &descriptorSetLayoutMat,
    };
    if (!Check(vkAllocateDescriptorSets(device, &descMatAI, &descriptorSetMat)))
    {
        return false;
    }

    return true;
}

bool VulkanAdapter::InitVulkanSkyboxDescriptorSetLayout()
{
    // set = 0, binding = 0 : SamplerCube (combined image sampler)
    VkDescriptorSetLayoutBinding binding{
        .binding         = 0,
        .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT};
    VkDescriptorSetLayoutCreateInfo layoutCI{
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings    = &binding};
    if (!Check(vkCreateDescriptorSetLayout(device, &layoutCI, nullptr, &descriptorSetLayoutSky)))
    {
        return false;
    }

    VkDescriptorPoolSize poolSize{
        .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1};
    VkDescriptorPoolCreateInfo poolCI{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = 1,
        .poolSizeCount = 1,
        .pPoolSizes    = &poolSize};
    if (!Check(vkCreateDescriptorPool(device, &poolCI, nullptr, &descriptorPoolSky)))
    {
        return false;
    }

    VkDescriptorSetAllocateInfo allocAI{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = descriptorPoolSky,
        .descriptorSetCount = 1,
        .pSetLayouts        = &descriptorSetLayoutSky};
    if (!Check(vkAllocateDescriptorSets(device, &allocAI, &descriptorSetSky)))
    {
        return false;
    }
    return true;
}

bool VulkanAdapter::InitVulkanLinePipeline()
{
    // pipeline layout: push constant only (FrameUniforms addr + optional index)
    VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset     = 0,
        .size       = sizeof(PushConstant)};

    VkPipelineLayoutCreateInfo pipelineLayoutCI{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 0,
        .pSetLayouts            = nullptr,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstantRange};

    if (!Check(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &linePipelineLayout)))
    {
        return false;
    }

    std::vector<VkPipelineShaderStageCreateInfo> shaderStagesCIs{
        VkPipelineShaderStageCreateInfo{
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_VERTEX_BIT,
            .module = lineShader,
            .pName  = "main"},
        VkPipelineShaderStageCreateInfo{
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = lineShader,
            .pName  = "main"}};

    VkVertexInputBindingDescription vertexBinding{
        .binding   = 0,
        .stride    = sizeof(Line),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};

    std::vector<VkVertexInputAttributeDescription> vertexAttributes{
        VkVertexInputAttributeDescription{
            .location = 0,
            .binding  = 0,
            .format   = VK_FORMAT_R32G32B32_SFLOAT,
            .offset   = offsetof(Line, pos)},
        VkVertexInputAttributeDescription{
            .location = 1,
            .binding  = 0,
            .format   = VK_FORMAT_R32G32B32A32_SFLOAT,
            .offset   = offsetof(Line, color)}};

    VkPipelineVertexInputStateCreateInfo vertexInputStageCI{
        .sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount   = 1,
        .pVertexBindingDescriptions      = &vertexBinding,
        .vertexAttributeDescriptionCount = (uint32_t)vertexAttributes.size(),
        .pVertexAttributeDescriptions    = vertexAttributes.data()};

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{
        .sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST};

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
        .sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode    = VK_CULL_MODE_NONE,
        .frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth   = 1.0f};

    VkPipelineMultisampleStateCreateInfo multisampleStateCI{
        .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};

    VkPipelineDepthStencilStateCreateInfo depthStencilStateCI{
        .sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable  = VK_TRUE,
        .depthWriteEnable = VK_FALSE,
        .depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL};

    VkPipelineColorBlendAttachmentState blendAttachment{
        .blendEnable    = VK_FALSE,
        .colorWriteMask = 0xF};

    VkPipelineColorBlendStateCreateInfo colorBlendStateCI{
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments    = &blendAttachment};

    const VkFormat                imageFormat = VK_FORMAT_B8G8R8A8_SRGB;
    const VkFormat                depthFormat = VK_FORMAT_D24_UNORM_S8_UINT;
    VkPipelineRenderingCreateInfo renderingCI{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &imageFormat,
        .depthAttachmentFormat   = depthFormat};

    VkGraphicsPipelineCreateInfo graphicsPipelineCI{
        .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext               = &renderingCI,
        .stageCount          = (uint32_t)shaderStagesCIs.size(),
        .pStages             = shaderStagesCIs.data(),
        .pVertexInputState   = &vertexInputStageCI,
        .pInputAssemblyState = &inputAssemblyStateCI,
        .pViewportState      = &viewportStateCI,
        .pRasterizationState = &rasterizationStateCI,
        .pMultisampleState   = &multisampleStateCI,
        .pDepthStencilState  = &depthStencilStateCI,
        .pColorBlendState    = &colorBlendStateCI,
        .pDynamicState       = &dynamicStateCI,
        .layout              = linePipelineLayout};

    if (!Check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphicsPipelineCI, nullptr, &linePipeline)))
    {
        return false;
    }

    return true;
}

bool VulkanAdapter::InitVulkanMeshPipeline()
{
    std::array<VkDescriptorSetLayout, 2> setLayouts{
        descriptorSetLayoutTex, // set=0 : textures[]
        descriptorSetLayoutMat  // set=1 : material SSBO
    };
    VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .size       = sizeof(PushConstant)};
    VkPipelineLayoutCreateInfo pipelineLayoutCI{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = (uint32_t)setLayouts.size(),
        .pSetLayouts            = setLayouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstantRange};
    if (!Check(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &meshPipelineLayout)))
    {
        return false;
    }

    std::vector<VkPipelineShaderStageCreateInfo> shaderStagesCIs{
        VkPipelineShaderStageCreateInfo{
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_VERTEX_BIT,
            .module = meshShader,
            .pName  = "main"},
        VkPipelineShaderStageCreateInfo{
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = meshShader,
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
            .offset   = offsetof(Vertex, uv)},
        VkVertexInputAttributeDescription{
            .location = 3,
            .binding  = 0,
            .format   = VK_FORMAT_R32G32B32A32_SFLOAT,
            .offset   = offsetof(Vertex, color)},
        VkVertexInputAttributeDescription{
            .location = 4,
            .binding  = 0,
            .format   = VK_FORMAT_R32G32B32A32_SFLOAT,
            .offset   = offsetof(Vertex, tangent)}};
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
        .layout              = meshPipelineLayout};
    if (!Check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphicsPipelineCI, nullptr, &meshPipeline)))
    {
        return false;
    }
    return true;
}

bool VulkanAdapter::InitVulkanSkyboxPipeline()
{
    // pipeline layout: [ set0: skybox cubemap ] + push constant(frameUniformsAddr)
    VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset     = 0,
        .size       = sizeof(PushConstant)};
    VkPipelineLayoutCreateInfo pipelineLayoutCI{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &descriptorSetLayoutSky,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstantRange};
    if (!Check(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &skyboxPipelineLayout)))
    {
        return false;
    }

    std::vector<VkPipelineShaderStageCreateInfo> shaderStagesCIs{
        VkPipelineShaderStageCreateInfo{
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_VERTEX_BIT,
            .module = skyboxShader,
            .pName  = "main"},
        VkPipelineShaderStageCreateInfo{
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = skyboxShader,
            .pName  = "main"}};
    VkVertexInputBindingDescription vertexBinding{
        .binding   = 0,
        .stride    = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
    std::vector<VkVertexInputAttributeDescription> vertexAttributes{
        VkVertexInputAttributeDescription{
            .location = 0,
            .binding  = 0,
            .format   = VK_FORMAT_R32G32B32_SFLOAT}};
    VkPipelineVertexInputStateCreateInfo vertexInputStageCI{
        .sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount   = 1,
        .pVertexBindingDescriptions      = &vertexBinding,
        .vertexAttributeDescriptionCount = 1,
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
        .sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode    = VK_CULL_MODE_FRONT_BIT,
        .frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth   = 1.0f};
    VkPipelineMultisampleStateCreateInfo multisampleStateCI{
        .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineDepthStencilStateCreateInfo depthStencilStateCI{
        .sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable  = VK_TRUE,
        .depthWriteEnable = VK_FALSE,
        .depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL};
    VkPipelineColorBlendAttachmentState blendAttachment{
        .blendEnable    = VK_FALSE,
        .colorWriteMask = 0xF};
    VkPipelineColorBlendStateCreateInfo colorBlendStateCI{
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments    = &blendAttachment};
    const VkFormat                imageFormat = VK_FORMAT_B8G8R8A8_SRGB;
    const VkFormat                depthFormat = VK_FORMAT_D24_UNORM_S8_UINT;
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
        .layout              = skyboxPipelineLayout,
    };
    if (!Check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphicsPipelineCI, nullptr, &skyboxPipeline)))
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
                Node node;
                if (m_parser.Parse(data.file, node))
                {
                    LoadNode(node);
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
                    glm::vec3 offset = camPosition - navigation;
                    float     radius = glm::length(offset);
                    if (radius < 1e-6f)
                    {
                        radius = 1.0f;
                    }

                    float sensitivity = 0.005f;
                    camRotation.y -= data.deltaX * sensitivity;
                    camRotation.x += data.deltaY * sensitivity;

                    constexpr float limit = glm::half_pi<float>() - 0.001f;
                    camRotation.x         = std::clamp(camRotation.x, -limit, +limit);

                    float pitch = camRotation.x;
                    float yaw   = camRotation.y;

                    glm::vec3 dir;
                    dir.x = std::cosf(pitch) * std::sinf(yaw);
                    dir.y = std::sinf(pitch);
                    dir.z = std::cosf(pitch) * std::cosf(yaw);

                    camPosition = navigation + dir * radius;
                }
                else if (data.rightBtnPressing)
                {
                    glm::vec3 offset = camPosition - navigation;
                    float     radius = glm::length(offset);
                    if (radius < 1e-6f)
                    {
                        radius = 1.0f;
                    }

                    float pitch = camRotation.x;
                    float yaw   = camRotation.y;

                    glm::vec3 dir;
                    dir.x = std::cosf(pitch) * std::sinf(yaw);
                    dir.y = std::sinf(pitch);
                    dir.z = std::cosf(pitch) * std::cosf(yaw);
                    dir   = glm::normalize(dir);

                    glm::vec3 forward = -dir;
                    glm::vec3 worldUp = glm::vec3(0, 1, 0);

                    glm::vec3   right  = glm::normalize(glm::cross(forward, worldUp));
                    glm::vec3   up     = glm::normalize(glm::cross(right, forward));
                    const float k      = 0.1f;
                    float       length = radius * k;

                    float dx = data.deltaX * elapsed * length;
                    float dy = data.deltaY * elapsed * length;

                    glm::vec3 pan = (-right * dx) + (up * dy);

                    camPosition += pan;
                    navigation += pan;
                }
            }
            else if (data.event == MouseEventType::Wheel)
            {
                glm::vec3   gazeDir = camPosition - navigation;
                glm::vec3   dir     = glm::normalize(gazeDir);
                float       dist    = glm::length(gazeDir);
                const float minDist = 0.02f;
                const float maxDist = 1e6f;
                float       steps   = data.deltaY / 120.0f;
                const float k       = 0.15f;
                float       newDist = dist * std::pow(1.0f - k, steps);
                newDist             = std::clamp(newDist, minDist, maxDist);
                camPosition         = navigation + dir * newDist;
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
    for (uint32_t i = 0; i < imageCount; ++i)
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

void VulkanAdapter::CollectMeshes(const Node& node, std::vector<const Mesh*>& out)
{
    for (const Mesh& mesh : node.meshes)
    {
        out.push_back(&mesh);
    }

    for (const Node& child : node.children)
    {
        CollectMeshes(child, out);
    }
}

void VulkanAdapter::LoadNode(const Node& node)
{
    std::lock_guard<std::mutex> guard(renderLock);

    vkDeviceWaitIdle(device);

    std::vector<const Mesh*> meshes;
    CollectMeshes(node, meshes);

    std::vector<MeshGpuBuffer>         newMeshBuffers;
    std::vector<TextureResource>       newTextures;
    std::vector<VkDescriptorImageInfo> textureDescriptors;
    std::vector<MeshUniforms>          meshUniformsVec;

    std::unordered_map<std::string, int> texIndexCache;
    texIndexCache.reserve(meshes.size());
    auto FindOrLoadTexture = [&](const Texture& texture) -> int {
        int diffuseTexIndex = -1;
        if (!texture.id.empty())
        {
            auto it = texIndexCache.find(texture.id);
            if (it != texIndexCache.end())
            {
                diffuseTexIndex = it->second;
            }
            else
            {
                if (LoadTexture(texture, newTextures))
                {
                    diffuseTexIndex = static_cast<int>(textureDescriptors.size());
                    texIndexCache.emplace(texture.id, diffuseTexIndex);

                    textureDescriptors.emplace_back(VkDescriptorImageInfo{
                        .sampler     = newTextures.back().sampler,
                        .imageView   = newTextures.back().view,
                        .imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL});
                }
            }
        }
        return diffuseTexIndex;
    };

    for (const Mesh* mesh : meshes)
    {
        if (!LoadMesh(*mesh, newMeshBuffers))
        {
            continue;
        }

        int diffuseTexIndex  = FindOrLoadTexture(mesh->material.diffuseTexture);
        int emissiveTexIndex = FindOrLoadTexture(mesh->material.emissiveTexture);
        int normalTexIndex   = FindOrLoadTexture(mesh->material.normalTexture);
        int ormTexIndex      = FindOrLoadTexture(mesh->material.ormTexture);

        MeshUniforms meshUniforms;
        meshUniforms.model            = mesh->trans;
        meshUniforms.diffuseColor     = mesh->material.diffuseColor;
        meshUniforms.emissiveColor    = mesh->material.emissiveColor;
        meshUniforms.textureIndexes.x = diffuseTexIndex;
        meshUniforms.textureIndexes.y = emissiveTexIndex;
        meshUniforms.textureIndexes.z = normalTexIndex;
        meshUniforms.textureIndexes.w = ormTexIndex;

        meshUniformsVec.emplace_back(std::move(meshUniforms));
    }

    navigation  = node.aabb.Center();
    camPosition = navigation + glm::vec3(0, 0, node.aabb.Length().z * 2.5f);
    camRotation = glm::quat(1, 0, 0, 0);

    // swap & destroy old resources
    std::swap(meshBuffers, newMeshBuffers);
    for (const MeshGpuBuffer& bufferDesc : newMeshBuffers)
    {
        vmaDestroyBuffer(allocator, bufferDesc.buffer, bufferDesc.allocation);
    }

    std::swap(meshTextures, newTextures);
    for (const TextureResource& texture : newTextures)
    {
        vkDestroyImageView(device, texture.view, nullptr);
        vkDestroySampler(device, texture.sampler, nullptr);
        vmaDestroyImage(allocator, texture.image, texture.allocation);
    }

    // update texture descriptor set
    if (!textureDescriptors.empty())
    {
        VkWriteDescriptorSet writeDescSetTex{
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = descriptorSetTex,
            .dstBinding      = 0,
            .descriptorCount = static_cast<uint32_t>(textureDescriptors.size()),
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo      = textureDescriptors.data()};
        vkUpdateDescriptorSets(device, 1, &writeDescSetTex, 0, nullptr);
    }

    // update material ssbo
    VkDeviceSize meshUniformSize = VkDeviceSize(meshUniformsVec.size()) * sizeof(MeshUniforms);
    if (nullptr != meshUniformsGpuBuffer.mapped)
    {
        vmaUnmapMemory(allocator, meshUniformsGpuBuffer.allocation);
        meshUniformsGpuBuffer.mapped = nullptr;
    }
    if (VK_NULL_HANDLE != meshUniformsGpuBuffer.buffer)
    {
        vmaDestroyBuffer(allocator, meshUniformsGpuBuffer.buffer, meshUniformsGpuBuffer.allocation);
    }

    VkBufferCreateInfo bufferCI{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = meshUniformSize,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    VmaAllocationCreateInfo bufferAllocCI{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO};
    VmaAllocationInfo bufferAI{};
    if (!Check(vmaCreateBuffer(allocator, &bufferCI, &bufferAllocCI, &meshUniformsGpuBuffer.buffer, &meshUniformsGpuBuffer.allocation, &bufferAI)))
    {
        return;
    }
    vmaMapMemory(allocator, meshUniformsGpuBuffer.allocation, &meshUniformsGpuBuffer.mapped);
    std::memcpy(meshUniformsGpuBuffer.mapped, meshUniformsVec.data(), meshUniformSize);
    vmaFlushAllocation(allocator, meshUniformsGpuBuffer.allocation, 0, meshUniformSize);

    VkDescriptorBufferInfo descBI{
        .buffer = meshUniformsGpuBuffer.buffer,
        .offset = 0,
        .range  = meshUniformSize,
    };
    VkWriteDescriptorSet writeDescSetMat{
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet          = descriptorSetMat,
        .dstBinding      = 0,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo     = &descBI,
    };
    vkUpdateDescriptorSets(device, 1, &writeDescSetMat, 0, nullptr);
}

bool VulkanAdapter::LoadMesh(const Mesh& mesh, std::vector<MeshGpuBuffer>& buffers)
{
    MeshGpuBuffer      bufferDesc;
    VkDeviceSize       vbSize = sizeof(Vertex) * mesh.vertices.size();
    VkDeviceSize       ibSize = sizeof(Index) * mesh.indices.size();
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
    if (Check(vmaCreateBuffer(allocator, &bufferCI, &bufferAllocCI, &bufferDesc.buffer, &bufferDesc.allocation, nullptr)))
    {
        bufferDesc.offsetOfIndexBuffer = vbSize;
        bufferDesc.indexCount          = (Index)mesh.indices.size();

        void* mapped = nullptr;
        vmaMapMemory(allocator, bufferDesc.allocation, &mapped);
        memcpy(mapped, mesh.vertices.data(), vbSize);
        memcpy(((char*)mapped) + vbSize, mesh.indices.data(), ibSize);
        vmaUnmapMemory(allocator, bufferDesc.allocation);

        buffers.emplace_back(std::move(bufferDesc));
        return true;
    }

    return false;
}

bool VulkanAdapter::LoadTexture(const Texture& tex, std::vector<TextureResource>& textures)
{
    if (tex.id.empty() || tex.width == 0 || tex.height == 0)
    {
        return false;
    }

    TextureResource         texResource;
    VkImageCreateInfo       texImgCI{.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                     .imageType   = VK_IMAGE_TYPE_2D,
                                     .format      = VK_FORMAT_R8G8B8A8_UNORM,
                                     .extent      = {.width  = (uint32_t)tex.width,
                                                     .height = (uint32_t)tex.height,
                                                     .depth  = 1},
                                     .mipLevels   = 1,
                                     .arrayLayers = 1,
                                     .samples     = VK_SAMPLE_COUNT_1_BIT,
                                     .tiling      = VK_IMAGE_TILING_OPTIMAL,
                                     .usage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                        VK_IMAGE_USAGE_SAMPLED_BIT,
                                     .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo texImageAllocCI{.usage = VMA_MEMORY_USAGE_AUTO};
    if (!Check(vmaCreateImage(allocator, &texImgCI, &texImageAllocCI, &texResource.image, &texResource.allocation, nullptr)))
    {
        return false;
    }

    VkImageViewCreateInfo texViewCI{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image            = texResource.image,
        .viewType         = VK_IMAGE_VIEW_TYPE_2D,
        .format           = texImgCI.format,
        .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                             .levelCount = 1,
                             .layerCount = 1}};
    if (!Check(vkCreateImageView(device, &texViewCI, nullptr, &texResource.view)))
    {
        return false;
    }

    // upload
    size_t             size             = tex.width * tex.height * tex.channels;
    VkBuffer           imgSrcBuffer     = VK_NULL_HANDLE;
    VmaAllocation      imgSrcAllocation = VK_NULL_HANDLE;
    VkBufferCreateInfo imgSrcBufferCI{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = (uint32_t)size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
    VmaAllocationCreateInfo imgSrcAllocCI{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO};
    if (!Check(vmaCreateBuffer(allocator, &imgSrcBufferCI, &imgSrcAllocCI, &imgSrcBuffer, &imgSrcAllocation, nullptr)))
    {
        return false;
    }

    void* imgSrcBufferPtr = nullptr;
    if (!Check(vmaMapMemory(allocator, imgSrcAllocation, &imgSrcBufferPtr)))
    {
        return false;
    }
    memcpy(imgSrcBufferPtr, tex.buffer.data(), size);

    VkFence           fenceOneTime = VK_NULL_HANDLE;
    VkFenceCreateInfo fenceOneTimeCI{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (!Check(vkCreateFence(device, &fenceOneTimeCI, nullptr, &fenceOneTime)))
    {
        return false;
    }

    VkCommandBuffer             cbOneTime = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cbOneTimeCI{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = commandPool,
        .commandBufferCount = 1};
    if (!Check(vkAllocateCommandBuffers(device, &cbOneTimeCI, &cbOneTime)))
    {
        return false;
    }

    VkCommandBufferBeginInfo cbOneTimeBI{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if (!Check(vkBeginCommandBuffer(cbOneTime, &cbOneTimeBI)))
    {
        return false;
    }

    VkImageMemoryBarrier2 barrierTexImage{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_2_NONE,
        .srcAccessMask = VK_ACCESS_2_NONE,
        .dstStageMask  = VK_PIPELINE_STAGE_TRANSFER_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .image         = texResource.image,
        .subresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                          .levelCount = 1,
                          .layerCount = 1}};
    VkDependencyInfo barrierTexInfo{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &barrierTexImage};
    vkCmdPipelineBarrier2(cbOneTime, &barrierTexInfo);

    std::vector<VkBufferImageCopy> copyRegions;
    int                            levels = 1;
    for (int i = 0; i < levels; ++i)
    {
        VkDeviceAddress mipOffset = 0;

        VkBufferImageCopy bufferImageCopy{
            .bufferOffset = mipOffset,
            .imageSubresource{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel   = (uint32_t)i,
                .layerCount = 1},
            .imageExtent{.width  = (uint32_t)tex.width >> i,
                         .height = (uint32_t)tex.height >> i,
                         .depth  = 1}};

        copyRegions.emplace_back(std::move(bufferImageCopy));
    }
    VkImageLayout layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    vkCmdCopyBufferToImage(cbOneTime, imgSrcBuffer, texResource.image, layout, (uint32_t)copyRegions.size(), copyRegions.data());

    VkImageMemoryBarrier2 barrierTexRead{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout     = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
        .image         = texResource.image,
        .subresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                          .levelCount = 1,
                          .layerCount = 1}};
    barrierTexInfo.pImageMemoryBarriers = &barrierTexRead;
    vkCmdPipelineBarrier2(cbOneTime, &barrierTexInfo);

    if (!Check(vkEndCommandBuffer(cbOneTime)))
    {
        return false;
    }

    VkSubmitInfo oneTimeSI{
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &cbOneTime};
    if (!Check(vkQueueSubmit(queue, 1, &oneTimeSI, fenceOneTime)))
    {
        return false;
    }

    if (!Check(vkWaitForFences(device, 1, &fenceOneTime, VK_TRUE, UINT64_MAX)))
    {
        return false;
    }

    vkDestroyFence(device, fenceOneTime, nullptr);
    vmaUnmapMemory(allocator, imgSrcAllocation);
    vmaDestroyBuffer(allocator, imgSrcBuffer, imgSrcAllocation);
    vkFreeCommandBuffers(device, commandPool, 1, &cbOneTime);

    // sampler
    VkSamplerCreateInfo samplerCI{
        .sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter        = VK_FILTER_LINEAR,
        .minFilter        = VK_FILTER_LINEAR,
        .mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .anisotropyEnable = VK_TRUE,
        .maxAnisotropy    = 8.0f,
        .maxLod           = 1};
    if (!Check(vkCreateSampler(device, &samplerCI, nullptr, &texResource.sampler)))
    {
        return false;
    }

    textures.emplace_back(std::move(texResource));
    return true;
}

bool VulkanAdapter::GenerateAxisGridBuffer()
{
    std::vector<Line> v;

    // axis
    const float axisLen = 1000.0f;

    // grid
    const float gridSize = 2000.0f;
    const float half     = gridSize * 0.5f; // 1000
    const float step     = 10.0f;

    const glm::vec4 colorX = glm::vec4(1, 0, 0, 1);
    const glm::vec4 colorY = glm::vec4(0, 1, 0, 1);
    const glm::vec4 colorZ = glm::vec4(0, 0, 1, 1);
    const glm::vec4 colorG = glm::vec4(0.35f, 0.35f, 0.35f, 1.0f);

    // grid lines
    const int n = int(half / step);
    v.reserve(6 + (size_t)(n * 2) * 4 + 4);
    for (int i = -n; i <= n; ++i)
    {
        if (i == 0)
        {
            continue;
        }

        float p = i * step;

        // x = p : z from -half..+half
        v.push_back(Line{glm::vec3(p, 0.0f, -half), colorG});
        v.push_back(Line{glm::vec3(p, 0.0f, +half), colorG});

        // z = p : x from -half..+half
        v.push_back(Line{glm::vec3(-half, 0.0f, p), colorG});
        v.push_back(Line{glm::vec3(+half, 0.0f, p), colorG});
    }

    // negative-half of the two main axis lines
    // x = 0, z in [-half, 0]
    v.push_back(Line{glm::vec3(0.0f, 0.0f, -half), colorG});
    v.push_back(Line{glm::vec3(0.0f, 0.0f, 0.0f), colorG});

    // z = 0, x in [-half, 0]
    v.push_back(Line{glm::vec3(-half, 0.0f, 0.0f), colorG});
    v.push_back(Line{glm::vec3(0.0f, 0.0f, 0.0f), colorG});

    // axes
    v.push_back(Line{glm::vec3(0, 0, 0), colorX});
    v.push_back(Line{glm::vec3(+axisLen, 0, 0), colorX});

    v.push_back(Line{glm::vec3(0, 0, 0), colorY});
    v.push_back(Line{glm::vec3(0, +axisLen, 0), colorY});

    v.push_back(Line{glm::vec3(0, 0, 0), colorZ});
    v.push_back(Line{glm::vec3(0, 0, +axisLen), colorZ});

    // upload
    VkDeviceSize       size = VkDeviceSize(v.size() * sizeof(Line));
    VkBufferCreateInfo bufferCI{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = size,
        .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT};
    VmaAllocationCreateInfo allocCI{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO};
    if (!Check(vmaCreateBuffer(allocator, &bufferCI, &allocCI, &axisGridBuffer.buffer, &axisGridBuffer.allocation, nullptr)))
    {
        return false;
    }

    void* mapped = nullptr;
    vmaMapMemory(allocator, axisGridBuffer.allocation, &mapped);
    std::memcpy(mapped, v.data(), (size_t)size);
    vmaUnmapMemory(allocator, axisGridBuffer.allocation);

    axisGridBuffer.indexCount = (uint32_t)v.size(); // vertexCount
    return true;
}

bool VulkanAdapter::GenerateSkyboxBuffer()
{
    std::array<Vertex, 8> vertices{};
    vertices[0].pos = glm::vec3(-1, -1, -1);
    vertices[1].pos = glm::vec3(+1, -1, -1);
    vertices[2].pos = glm::vec3(+1, +1, -1);
    vertices[3].pos = glm::vec3(-1, +1, -1);
    vertices[4].pos = glm::vec3(-1, -1, +1);
    vertices[5].pos = glm::vec3(+1, -1, +1);
    vertices[6].pos = glm::vec3(+1, +1, +1);
    vertices[7].pos = glm::vec3(-1, +1, +1);

    std::array<Index, 36> indices = {
        // -Z (back)
        0, 2, 1,
        0, 3, 2,

        // +Z (front)
        4, 5, 6,
        4, 6, 7,

        // -X (left)
        0, 7, 3,
        0, 4, 7,

        // +X (right)
        1, 2, 6,
        1, 6, 5,

        // -Y (bottom)
        0, 1, 5,
        0, 5, 4,

        // +Y (top)
        3, 6, 2,
        3, 7, 6};

    VkDeviceSize vbSize = VkDeviceSize(vertices.size() * sizeof(Vertex));
    VkDeviceSize ibSize = VkDeviceSize(indices.size() * sizeof(Index));

    VkBufferCreateInfo bufferCI{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = vbSize + ibSize,
        .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT};

    VmaAllocationCreateInfo allocCI{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO};

    if (!Check(vmaCreateBuffer(allocator, &bufferCI, &allocCI, &skyboxBuffer.buffer, &skyboxBuffer.allocation, nullptr)))
    {
        return false;
    }

    skyboxBuffer.offsetOfIndexBuffer = vbSize;
    skyboxBuffer.indexCount          = (uint32_t)indices.size();

    void* mapped = nullptr;
    vmaMapMemory(allocator, skyboxBuffer.allocation, &mapped);
    std::memcpy(mapped, vertices.data(), (size_t)vbSize);
    std::memcpy((char*)mapped + vbSize, indices.data(), (size_t)ibSize);
    vmaUnmapMemory(allocator, skyboxBuffer.allocation);

    return true;
}

bool VulkanAdapter::LoadSkybox(const std::string& path)
{
    ktxTexture2*   ktx2 = nullptr;
    KTX_error_code kres = ktxTexture2_CreateFromNamedFile(path.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktx2);
    if (kres != KTX_SUCCESS || nullptr == ktx2)
    {
        return false;
    }

    if (ktx2->numFaces != 6)
    {
        ktxTexture_Destroy(ktxTexture(ktx2));
        return false;
    }

    if (ktxTexture2_NeedsTranscoding(ktx2))
    {
        if (ktxTexture2_TranscodeBasis(ktx2, KTX_TTF_RGBA32, 0) != KTX_SUCCESS)
        {
            ktxTexture_Destroy(ktxTexture(ktx2));
            return false;
        }
    }

    ktxTexture*    ktx       = ktxTexture(ktx2);
    const uint32_t width     = ktx->baseWidth;
    const uint32_t height    = ktx->baseHeight;
    const uint32_t mipLevels = ktx->numLevels;
    const VkFormat format    = (VkFormat)ktx2->vkFormat;

    if (width == 0 || height == 0 || mipLevels == 0 || format == VK_FORMAT_UNDEFINED || ktx->dataSize == 0 || ktx->pData == nullptr)
    {
        ktxTexture_Destroy(ktx);
        return false;
    }

    TextureResource   newSkybox;
    VkImageCreateInfo imgCI{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags         = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = format,
        .extent        = {width, height, 1},
        .mipLevels     = mipLevels,
        .arrayLayers   = 6,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo imgAllocCI{.usage = VMA_MEMORY_USAGE_AUTO};
    if (!Check(vmaCreateImage(allocator, &imgCI, &imgAllocCI, &newSkybox.image, &newSkybox.allocation, nullptr)))
    {
        ktxTexture_Destroy(ktx);
        return false;
    }

    auto Destroy = [](VkDevice device, VmaAllocator allocator, TextureResource& r) {
        if (r.view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device, r.view, nullptr);
            r.view = VK_NULL_HANDLE;
        }
        if (r.sampler != VK_NULL_HANDLE)
        {
            vkDestroySampler(device, r.sampler, nullptr);
            r.sampler = VK_NULL_HANDLE;
        }
        if (r.image != VK_NULL_HANDLE)
        {
            vmaDestroyImage(allocator, r.image, r.allocation);
            r.image      = VK_NULL_HANDLE;
            r.allocation = VK_NULL_HANDLE;
        }
    };

    const VkDeviceSize dataSize     = (VkDeviceSize)ktx->dataSize;
    VkBuffer           stagingBuf   = VK_NULL_HANDLE;
    VmaAllocation      stagingAlloc = VK_NULL_HANDLE;
    VkBufferCreateInfo stagingBCI{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = dataSize,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
    VmaAllocationCreateInfo stagingACI{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO};
    if (!Check(vmaCreateBuffer(allocator, &stagingBCI, &stagingACI, &stagingBuf, &stagingAlloc, nullptr)))
    {
        Destroy(device, allocator, newSkybox);
        ktxTexture_Destroy(ktx);
        return false;
    }

    void* mapped = nullptr;
    if (!Check(vmaMapMemory(allocator, stagingAlloc, &mapped)))
    {
        vmaDestroyBuffer(allocator, stagingBuf, stagingAlloc);
        Destroy(device, allocator, newSkybox);
        ktxTexture_Destroy(ktx);
        return false;
    }
    std::memcpy(mapped, ktx->pData, (size_t)dataSize);
    vmaUnmapMemory(allocator, stagingAlloc);

    VkFence           fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fenceCI{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (!Check(vkCreateFence(device, &fenceCI, nullptr, &fence)))
    {
        vmaDestroyBuffer(allocator, stagingBuf, stagingAlloc);
        Destroy(device, allocator, newSkybox);
        ktxTexture_Destroy(ktx);
        return false;
    }

    VkCommandBuffer             cb = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cbAI{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = commandPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1};
    if (!Check(vkAllocateCommandBuffers(device, &cbAI, &cb)))
    {
        vkDestroyFence(device, fence, nullptr);
        vmaDestroyBuffer(allocator, stagingBuf, stagingAlloc);
        Destroy(device, allocator, newSkybox);
        ktxTexture_Destroy(ktx);
        return false;
    }

    VkCommandBufferBeginInfo cbBI{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if (!Check(vkBeginCommandBuffer(cb, &cbBI)))
    {
        vkFreeCommandBuffers(device, commandPool, 1, &cb);
        vkDestroyFence(device, fence, nullptr);
        vmaDestroyBuffer(allocator, stagingBuf, stagingAlloc);
        Destroy(device, allocator, newSkybox);
        ktxTexture_Destroy(ktx);
        return false;
    }

    VkImageMemoryBarrier2 toTransfer{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask     = VK_PIPELINE_STAGE_2_NONE,
        .srcAccessMask    = VK_ACCESS_2_NONE,
        .dstStageMask     = VK_PIPELINE_STAGE_TRANSFER_BIT,
        .dstAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .image            = newSkybox.image,
        .subresourceRange = {
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = mipLevels,
            .baseArrayLayer = 0,
            .layerCount     = 6}};
    VkDependencyInfo dep0{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &toTransfer};
    vkCmdPipelineBarrier2(cb, &dep0);

    std::vector<VkBufferImageCopy> regions;
    regions.reserve((size_t)mipLevels * 6);

    for (uint32_t level = 0; level < mipLevels; ++level)
    {
        const uint32_t w = std::max(1u, width >> level);
        const uint32_t h = std::max(1u, height >> level);

        for (uint32_t face = 0; face < 6; ++face)
        {
            ktx_size_t offset = 0;
            // (level, layer=0, faceSlice=face)
            if (ktxTexture_GetImageOffset(ktx, level, 0, face, &offset) != KTX_SUCCESS)
            {
                vkEndCommandBuffer(cb);
                vkFreeCommandBuffers(device, commandPool, 1, &cb);
                vkDestroyFence(device, fence, nullptr);
                vmaDestroyBuffer(allocator, stagingBuf, stagingAlloc);
                Destroy(device, allocator, newSkybox);
                ktxTexture_Destroy(ktx);
                return false;
            }

            VkBufferImageCopy r{
                .bufferOffset      = (VkDeviceSize)offset,
                .bufferRowLength   = 0,
                .bufferImageHeight = 0,
                .imageSubresource  = {
                     .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                     .mipLevel       = level,
                     .baseArrayLayer = face,
                     .layerCount     = 1},
                .imageOffset = {0, 0, 0},
                .imageExtent = {w, h, 1}};
            regions.push_back(r);
        }
    }

    vkCmdCopyBufferToImage(cb, stagingBuf, newSkybox.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, (uint32_t)regions.size(), regions.data());

    // TRANSFER_DST -> SHADER_READ
    VkImageMemoryBarrier2 toRead{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask     = VK_PIPELINE_STAGE_TRANSFER_BIT,
        .srcAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask     = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        .dstAccessMask    = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        .oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .image            = newSkybox.image,
        .subresourceRange = {
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = mipLevels,
            .baseArrayLayer = 0,
            .layerCount     = 6}};
    VkDependencyInfo dep1{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &toRead};
    vkCmdPipelineBarrier2(cb, &dep1);

    if (!Check(vkEndCommandBuffer(cb)))
    {
        vkFreeCommandBuffers(device, commandPool, 1, &cb);
        vkDestroyFence(device, fence, nullptr);
        vmaDestroyBuffer(allocator, stagingBuf, stagingAlloc);
        Destroy(device, allocator, newSkybox);
        ktxTexture_Destroy(ktx);
        return false;
    }

    VkSubmitInfo si{
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &cb};

    bool ok = Check(vkQueueSubmit(queue, 1, &si, fence)) &&
              Check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

    vkFreeCommandBuffers(device, commandPool, 1, &cb);
    vkDestroyFence(device, fence, nullptr);

    vmaDestroyBuffer(allocator, stagingBuf, stagingAlloc);
    ktxTexture_Destroy(ktx);

    if (!ok)
    {
        Destroy(device, allocator, newSkybox);
        return false;
    }

    VkImageViewCreateInfo viewCI{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image            = newSkybox.image,
        .viewType         = VK_IMAGE_VIEW_TYPE_CUBE,
        .format           = format,
        .subresourceRange = {
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = mipLevels,
            .baseArrayLayer = 0,
            .layerCount     = 6}};

    if (!Check(vkCreateImageView(device, &viewCI, nullptr, &newSkybox.view)))
    {
        Destroy(device, allocator, newSkybox);
        return false;
    }

    VkSamplerCreateInfo samplerCI{
        .sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter        = VK_FILTER_LINEAR,
        .minFilter        = VK_FILTER_LINEAR,
        .mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .anisotropyEnable = VK_TRUE,
        .maxAnisotropy    = 8.0f,
        .minLod           = 0.0f,
        .maxLod           = (float)mipLevels};

    if (!Check(vkCreateSampler(device, &samplerCI, nullptr, &newSkybox.sampler)))
    {
        Destroy(device, allocator, newSkybox);
        return false;
    }

    std::swap(skyboxTexture, newSkybox);
    Destroy(device, allocator, newSkybox);

    VkDescriptorImageInfo imgInfo{
        .sampler     = skyboxTexture.sampler,
        .imageView   = skyboxTexture.view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet write{
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet          = descriptorSetSky,
        .dstBinding      = 0,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo      = &imgInfo};
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    return true;
}
