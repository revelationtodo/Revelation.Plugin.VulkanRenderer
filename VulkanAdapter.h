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

struct alignas(16) FrameUniforms
{
    glm::mat4 projection = glm::mat4(1.0f);
    glm::mat4 view       = glm::mat4(1.0f);
    glm::vec4 lightDir   = glm::vec4(glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f)), 0.0f);
    glm::vec4 cameraPos  = glm::vec4(0.0f);
};

struct alignas(16) PointsUniforms
{
    glm::mat4 model        = glm::mat4(1.0f);
};

struct alignas(16) MeshUniforms
{
    glm::mat4 model = glm::mat4(1.0f);

    glm::vec4 diffuseColor  = glm::vec4(1.0f);
    glm::vec4 emissiveColor = glm::vec4(0.0f);

    // [diffuse, emissive, diffuse, orm]
    glm::ivec4 textureIndexes = glm::ivec4(-1);
};

struct alignas(16) PushConstant
{
    uint64_t frameUniformsAddr = 0;

    int64_t meshUniformsIndex   = -1;
    int64_t pointsUniformsIndex = -1;
};

struct MappedGpuBuffer
{
    VkBuffer        buffer        = VK_NULL_HANDLE;
    VmaAllocation   allocation    = VK_NULL_HANDLE;
    VkDeviceAddress deviceAddress = 0;
    void*           mapped        = nullptr;
};

struct PointsGpuBuffer
{
    VkBuffer      buffer      = VK_NULL_HANDLE;
    VmaAllocation allocation  = VK_NULL_HANDLE;
    Index         vertexCount = 0;
};

struct MeshGpuBuffer
{
    VkBuffer      buffer              = VK_NULL_HANDLE;
    VmaAllocation allocation          = VK_NULL_HANDLE;
    VkDeviceSize  offsetOfIndexBuffer = 0;
    Index         indexCount          = 0;
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
    bool InitVulkanSkyboxShader();
    bool InitVulkanPointShader();
    bool InitVulkanLineShader();
    bool InitVulkanMeshShader();
    bool InitVulkanSyncObjects();
    bool InitVulkanCommandPools();
    bool InitVulkanSkyboxDescriptorSetLayout();
    bool InitVulkanPointsDescriptorSetLayout();
    bool InitVulkanMeshDescriptorSetLayout();
    bool InitVulkanSkyboxPipeline();
    bool InitVulkanPointPipeline();
    bool InitVulkanLinePipeline();
    bool InitVulkanMeshPipeline();

    void PollInputEvents(double elapsed);

    bool UpdateSwapchain();
    void UpdateZnZf();

    void LoadNode(const Node& node);

    void LoadPoints(const Node& node);
    void CollectPoints(const Node& node, std::vector<const Points*>& out);
    bool LoadPoints(const Points& points, std::vector<PointsGpuBuffer>& buffers);

    void LoadMeshes(const Node& node);
    void CollectMeshes(const Node& node, std::vector<const Mesh*>& out);
    bool LoadMesh(const Mesh& mesh, std::vector<MeshGpuBuffer>& buffers);
    bool LoadTexture(const Texture& tex, std::vector<TextureResource>& textures);

    bool GenerateAxisGridBuffer();

    bool GenerateSkyboxBuffer();
    bool LoadSkybox(const std::string& path);

  private:
    VulkanRendererWidget* m_targetWindow = nullptr;

    Parser m_parser;
    bool   m_ready = false;

    //////////////////////////////////////////////////////////////////////////
    // Vulkan related
    using FrameUniformsGpuBuffers = std::array<MappedGpuBuffer, maxFramesInFlight>;
    using Fences                  = std::array<VkFence, maxFramesInFlight>;
    using PresentSemaphores       = std::array<VkSemaphore, maxFramesInFlight>;
    using RenderingSemaphores     = std::vector<VkSemaphore>;
    using CommandBuffers          = std::array<VkCommandBuffer, maxFramesInFlight>;
    using SlangGlobalSession      = Slang::ComPtr<slang::IGlobalSession>;

    MeshGpuBuffer axisGridBuffer;

    MeshGpuBuffer   skyboxBuffer;
    TextureResource skyboxTexture;

    std::vector<PointsGpuBuffer> pointsBuffers;
    MappedGpuBuffer              pointsUniformsGpuBuffer;

    std::vector<MeshGpuBuffer>   meshBuffers;
    std::vector<TextureResource> meshTextures;
    MappedGpuBuffer              meshUniformsGpuBuffer;

    glm::vec3 navigation  = glm::vec3(0.0f);
    glm::vec3 camPosition = glm::vec3(0.0f, -10.0f, 10.0f);
    float     zNear       = 1.0f;
    float     zFar        = 10000.0f;

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
    FrameUniformsGpuBuffers  frameUniformGpuBuffers;
    VkShaderModule           pointShader  = VK_NULL_HANDLE;
    VkShaderModule           lineShader   = VK_NULL_HANDLE;
    VkShaderModule           meshShader   = VK_NULL_HANDLE;
    VkShaderModule           skyboxShader = VK_NULL_HANDLE;
    SlangGlobalSession       slangGlobalSession;
    Fences                   fences;
    PresentSemaphores        presentSemaphores;
    RenderingSemaphores      renderSemaphores;
    VkCommandPool            commandPool = VK_NULL_HANDLE;
    CommandBuffers           commandBuffers;
    VkDescriptorSetLayout    descriptorSetLayoutSky = VK_NULL_HANDLE;
    VkDescriptorPool         descriptorPoolSky      = VK_NULL_HANDLE;
    VkDescriptorSet          descriptorSetSky       = VK_NULL_HANDLE;
    VkDescriptorSetLayout    descriptorSetLayoutPts = VK_NULL_HANDLE;
    VkDescriptorPool         descriptorPoolPts      = VK_NULL_HANDLE;
    VkDescriptorSet          descriptorSetPts       = VK_NULL_HANDLE;
    VkDescriptorSetLayout    descriptorSetLayoutTex = VK_NULL_HANDLE;
    VkDescriptorPool         descriptorPoolTex      = VK_NULL_HANDLE;
    VkDescriptorSet          descriptorSetTex       = VK_NULL_HANDLE;
    VkDescriptorSetLayout    descriptorSetLayoutMat = VK_NULL_HANDLE;
    VkDescriptorPool         descriptorPoolMat      = VK_NULL_HANDLE;
    VkDescriptorSet          descriptorSetMat       = VK_NULL_HANDLE;
    VkPipelineLayout         skyboxPipelineLayout   = VK_NULL_HANDLE;
    VkPipeline               skyboxPipeline         = VK_NULL_HANDLE;
    VkPipelineLayout         pointPipelineLayout    = VK_NULL_HANDLE;
    VkPipeline               pointPipeline          = VK_NULL_HANDLE;
    VkPipelineLayout         linePipelineLayout     = VK_NULL_HANDLE;
    VkPipeline               linePipeline           = VK_NULL_HANDLE;
    VkPipelineLayout         meshPipelineLayout     = VK_NULL_HANDLE;
    VkPipeline               meshPipeline           = VK_NULL_HANDLE;
    //////////////////////////////////////////////////////////////////////////
};