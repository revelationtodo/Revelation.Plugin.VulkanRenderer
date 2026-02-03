#pragma once
#include <set>
#include <string>
#include <glm/glm.hpp>

struct Vertex
{
    glm::vec3 pos    = glm::vec3(0);
    glm::vec3 normal = glm::vec3(0);
    glm::vec2 uv     = glm::vec2(0);
};

using Index = uint32_t;

struct AxisAlignedBox
{
    glm::vec3 min = glm::vec3(std::numeric_limits<float>::infinity());
    glm::vec3 max = glm::vec3(-std::numeric_limits<float>::infinity());

    glm::vec3 Center() const
    {
        return 0.5f * (min + max);
    }

    glm::vec3 Length() const
    {
        return max - min;
    }
};

struct Texture
{
    std::string          id       = "";
    int                  width    = 0;
    int                  height   = 0;
    int                  channels = 0;
    std::vector<uint8_t> buffer;
};

struct Mesh
{
    std::vector<Vertex> vertices;
    std::vector<Index>  indices;

    Texture diffuse;

    AxisAlignedBox aabb;

    glm::mat4 trans = glm::mat4(1);
};

struct Node
{
    std::vector<Node> children;
    std::vector<Mesh> meshes;

    AxisAlignedBox aabb;

    glm::mat4 trans = glm::mat4(1);
};

class Parser
{
  public:
    Parser();
    ~Parser();

    bool Parse(const std::string& file, Node& model);
};