#pragma once
#include <QWidget>

//////////////////////////////////////////////////////////////////////////
// rendering related headers
#include <vulkan/vulkan.h>
#define VOLK_IMPLEMENTATION
#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <volk/volk.h>
#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "slang/slang-com-ptr.h"
#include "slang/slang.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <ktx.h>
#include <ktxvulkan.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// global variables
constexpr uint32_t maxFramesInFlight{2};
//////////////////////////////////////////////////////////////////////////

class VulkanAdapter
{
  public:
    VulkanAdapter(QWidget* targetWindow);
    ~VulkanAdapter() = default;

    void Initialize();

  private:
    bool Check(VkResult result);
    bool Check(bool result);

    bool InitVulkanInstance();
    bool InitVulkanDevice();
    bool InitVulkanQueue();
    bool InitVmaAllocator();
    bool InitVulkanSwapchain();
    bool InitVulkanShader();
    bool InitVulkanSyncObjects();
    bool InitVulkanCommandPools();
    bool InitVulkanPipeline();

  private:
    QWidget* m_targetWindow = nullptr;

    //////////////////////////////////////////////////////////////////////////
    // Vulkan related
    struct ShaderData
    {
        glm::mat4 projection;
        glm::mat4 view;
        glm::mat4 model;
        glm::vec4 lightPos{0.0f, -10.0f, 10.0f, 0.0f};
    };

    struct ShaderDataBuffer
    {
        VmaAllocation   allocation    = VK_NULL_HANDLE;
        VkBuffer        buffer        = VK_NULL_HANDLE;
        VkDeviceAddress deviceAddress = {};
        void*           mapped        = nullptr;
    };

    using ShaderDataBuffers   = std::array<ShaderDataBuffer, maxFramesInFlight>;
    using Fences              = std::array<VkFence, maxFramesInFlight>;
    using PresentSemaphores   = std::array<VkSemaphore, maxFramesInFlight>;
    using RenderingSemaphores = std::vector<VkSemaphore>;
    using CommandBuffers      = std::array<VkCommandBuffer, maxFramesInFlight>;
    using SlangGlobalSession  = Slang::ComPtr<slang::IGlobalSession>;

    uint32_t                 imageIndex       = 0;
    uint32_t                 frameIndex       = 0;
    VkInstance               instance         = VK_NULL_HANDLE;
    VkPhysicalDevice         physicalDevice   = VK_NULL_HANDLE;
    uint32_t                 queueFamilyIndex = 0;
    VkDevice                 device           = VK_NULL_HANDLE;
    VkQueue                  queue            = VK_NULL_HANDLE;
    VmaAllocator             allocator        = VK_NULL_HANDLE;
    VkSurfaceKHR             surface          = VK_NULL_HANDLE;
    VkSwapchainKHR           swapchain        = VK_NULL_HANDLE;
    std::vector<VkImage>     swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    VkImage                  depthImage           = VK_NULL_HANDLE;
    VmaAllocation            depthImageAllocation = VK_NULL_HANDLE;
    VkImageView              depthImageView       = VK_NULL_HANDLE;
    ShaderDataBuffers        shaderDataBuffers;
    VkShaderModule           shaderModule = VK_NULL_HANDLE;
    Fences                   fences;
    PresentSemaphores        presentSemaphores;
    RenderingSemaphores      renderSemaphores;
    VkCommandPool            commandPool = VK_NULL_HANDLE;
    CommandBuffers           commandBuffers;
    SlangGlobalSession       slangGlobalSession;
    //////////////////////////////////////////////////////////////////////////
};