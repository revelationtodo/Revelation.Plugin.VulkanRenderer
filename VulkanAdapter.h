#pragma once
#include <QWidget>

#include <volk/volk.h>
#include <vma/vk_mem_alloc.h>

#include <array>
#include <string>
#include <vector>
#include <iostream>
#include <mutex>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "slang/slang-com-ptr.h"
#include "slang/slang.h"

#include <ktx.h>
#include <ktxvulkan.h>

#include "Parser/ParserManager.h"

class VulkanRendererWidget;

constexpr uint32_t maxFramesInFlight{2};

class VulkanAdapter
{
  public:
    VulkanAdapter(VulkanRendererWidget* targetWindow);
    ~VulkanAdapter();

    void Initialize();
    void Uninitialize();

    bool IsReady();
    void Tick(double delta);

  private:
    bool Check(VkResult result);
    bool Check(bool result);
    bool CheckSwapchain(VkResult result);

    bool InitVulkanInstance();
    bool InitVulkanDevice();
    bool InitVulkanQueue();
    bool InitVmaAllocator();
    bool InitVulkanSwapchain();
    bool InitVulkanShader();
    bool InitVulkanSyncObjects();
    bool InitVulkanCommandPools();
    bool InitVulkanDescriptorSetLayout();
    bool InitVulkanPipeline();

    void PollInputEvents();

    bool UpdateSwapchain();
    void LoadModel(const Model& model);

  private:
    VulkanRendererWidget* m_targetWindow = nullptr;
    ParserManager         m_parserManager;

    bool m_ready = false;

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

    struct Vertex
    {
        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec2 uv;
    };

    using ShaderDataBuffers   = std::array<ShaderDataBuffer, maxFramesInFlight>;
    using Fences              = std::array<VkFence, maxFramesInFlight>;
    using PresentSemaphores   = std::array<VkSemaphore, maxFramesInFlight>;
    using RenderingSemaphores = std::vector<VkSemaphore>;
    using CommandBuffers      = std::array<VkCommandBuffer, maxFramesInFlight>;
    using SlangGlobalSession  = Slang::ComPtr<slang::IGlobalSession>;

    struct BufferDesc
    {
        VkBuffer      buffer     = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;

        VkDeviceSize offsetOfIndexBuffer = 0;
        uint16_t     indexCount          = 0;
    };

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
    ShaderData               shaderData;
    ShaderDataBuffers        shaderDataBuffers;
    VkShaderModule           shaderModule = VK_NULL_HANDLE;
    SlangGlobalSession       slangGlobalSession;
    Fences                   fences;
    PresentSemaphores        presentSemaphores;
    RenderingSemaphores      renderSemaphores;
    VkCommandPool            commandPool = VK_NULL_HANDLE;
    CommandBuffers           commandBuffers;
    VkDescriptorSetLayout    descriptorSetLayoutTex = VK_NULL_HANDLE;
    VkDescriptorPool         descriptorPool         = VK_NULL_HANDLE;
    VkDescriptorSet          descriptorSetTex       = VK_NULL_HANDLE;
    VkPipelineLayout         pipelineLayout         = VK_NULL_HANDLE;
    VkPipeline               pipeline               = VK_NULL_HANDLE;

    std::vector<BufferDesc> modelBuffers;

    std::mutex renderLock;
    bool       updateSwapchain = false;
    //////////////////////////////////////////////////////////////////////////
};