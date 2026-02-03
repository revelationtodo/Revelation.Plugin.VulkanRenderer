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

#include "Parser/Parser.h"

class VulkanRendererWidget;

constexpr uint32_t maxFramesInFlight{2};

struct alignas(16) ShaderData
{
    glm::mat4 projection = glm::mat4(1);
    glm::mat4 view       = glm::mat4(1);
    glm::vec4 lightPos   = glm::vec4(0.0f, -1000.0f, 1000.0f, 0.0f);
    glm::vec4 cameraPos  = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
};

struct alignas(16) PushConstant
{
    uint64_t shaderDataAddr = 0;
    uint64_t _pad0          = 0;

    glm::mat4 model        = glm::mat4(1);
    int       textureIndex = -1;
};

struct ShaderDataBuffer
{
    VmaAllocation   allocation    = VK_NULL_HANDLE;
    VkBuffer        buffer        = VK_NULL_HANDLE;
    VkDeviceAddress deviceAddress = 0;
    void*           mapped        = nullptr;
};

struct BufferDesc
{
    VkBuffer      buffer     = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;

    VkDeviceSize offsetOfIndexBuffer = 0;
    Index        indexCount          = 0;
};

struct TextureResource
{
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImage       image      = VK_NULL_HANDLE;
    VkImageView   view       = VK_NULL_HANDLE;
    VkSampler     sampler    = VK_NULL_HANDLE;
};

class VulkanAdapter
{
  public:
    VulkanAdapter(VulkanRendererWidget* targetWindow);
    ~VulkanAdapter();

    void Initialize();
    void Uninitialize();

    bool IsReady();
    void Tick(double elapsed);

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

    void PollInputEvents(double elapsed);

    bool UpdateSwapchain();

    void CollectMeshes(const Node& node, std::vector<const Mesh*>& out);
    void LoadNode(const Node& node);
    bool LoadMesh(const Mesh& mesh, std::vector<BufferDesc>& buffers);
    bool LoadTexture(const Texture& tex, std::vector<TextureResource>& textures);

  private:
    VulkanRendererWidget* m_targetWindow = nullptr;
    Parser                m_parser;

    bool m_ready = false;

    //////////////////////////////////////////////////////////////////////////
    // Vulkan related
    using ShaderDataBuffers   = std::array<ShaderDataBuffer, maxFramesInFlight>;
    using Fences              = std::array<VkFence, maxFramesInFlight>;
    using PresentSemaphores   = std::array<VkSemaphore, maxFramesInFlight>;
    using RenderingSemaphores = std::vector<VkSemaphore>;
    using CommandBuffers      = std::array<VkCommandBuffer, maxFramesInFlight>;
    using SlangGlobalSession  = Slang::ComPtr<slang::IGlobalSession>;

    std::vector<BufferDesc>      modelBuffers;
    std::vector<glm::mat4>       modelMatrices;
    std::vector<TextureResource> textures;
    std::vector<int>             textureIndexes;

    glm::vec3 navigation  = glm::vec3(0);
    glm::vec3 camPosition = glm::vec3(0);
    glm::quat camRotation = glm::quat(1, 0, 0, 0);

    uint32_t   imageIndex = 0;
    uint32_t   frameIndex = 0;
    std::mutex renderLock;
    bool       updateSwapchain = false;

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
    //////////////////////////////////////////////////////////////////////////
};