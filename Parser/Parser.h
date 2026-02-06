#pragma once
#include <string>
#include <vector>
#include <limits>
#include <filesystem>

#include <glm/glm.hpp>
#include <assimp/scene.h>

struct Vertex
{
    glm::vec3 pos     = glm::vec3(0.0f);
    glm::vec3 normal  = glm::vec3(0.0f);
    glm::vec2 uv      = glm::vec2(0.0f);
    glm::vec4 color   = glm::vec4(1.0f);
    glm::vec4 tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
};

using Index = uint32_t;

struct AxisAlignedBox
{
    glm::vec3 min = glm::vec3(std::numeric_limits<float>::infinity());
    glm::vec3 max = glm::vec3(-std::numeric_limits<float>::infinity());

    glm::vec3 Center() const { return 0.5f * (min + max); }
    glm::vec3 Length() const { return max - min; }
};

struct Texture
{
    std::string          id;
    int                  width    = 0;
    int                  height   = 0;
    int                  channels = 0;
    std::vector<uint8_t> buffer;
};

enum class TextureType : uint8_t
{
    Diffuse,
    Emissive,
    Normal
};

struct Material
{
    // diffuse / base color
    glm::vec4 diffuseColor = glm::vec4(1.0f);
    Texture   diffuseTexture;

    // emissive
    glm::vec4 emissiveColor = glm::vec4(0.0f);
    Texture   emissiveTexture;

    // normal
    Texture normalTexture;
};

struct Mesh
{
    std::vector<Vertex> vertices;
    std::vector<Index>  indices;

    Material       material;
    AxisAlignedBox aabb;

    glm::mat4 trans = glm::mat4(1.0f);
};

struct Node
{
    std::vector<Node> children;
    std::vector<Mesh> meshes;

    AxisAlignedBox aabb;
    glm::mat4      trans = glm::mat4(1.0f);
};

class Parser
{
  public:
    Parser();
    ~Parser();

    bool Parse(const std::string& file, Node& model);

  private:
    // -------- material / texture --------
    std::string GetTextureRefByType(aiMaterial* mat, TextureType type) const;
    glm::vec4   GetColorByType(aiMaterial* mat, TextureType type) const;

    std::string JoinPath(const std::string& dir, const std::string& rel) const;
    bool        LoadTexture(const std::string& texPath, const aiScene* scene, Texture& tex) const;

    // -------- math / transform --------
    glm::mat4      ToGlm(const aiMatrix4x4& m) const;
    void           ExpandAabb(AxisAlignedBox& aabb, const glm::vec3& p) const;
    void           MergeAabb(AxisAlignedBox& dst, const AxisAlignedBox& src) const;
    AxisAlignedBox TransformAabb(const AxisAlignedBox& local, const glm::mat4& M) const;

    // -------- parsing --------
    void ProcessNodeRecursive(const aiScene* scene, const aiNode* ain, const std::string& assetDir,
                              const glm::mat4& parentGlobal, Node& outNode) const;
    bool ParseOneMesh(const aiScene* scene, const aiMesh* aimesh, const std::string& assetDir,
                      const glm::mat4& meshGlobalTrans, Mesh& outMesh, AxisAlignedBox& outMeshWorldAabb) const;
};
