#pragma once
#include <set>
#include <string>
#include <glm/glm.hpp>

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
};

using Index = uint32_t;

struct AxisAlignedBox
{
    glm::vec3 min = glm::vec3(std::numeric_limits<float>::infinity());
    glm::vec3 max = glm::vec3(-std::numeric_limits<float>::infinity());

    glm::vec3 Center()
    {
        return 0.5f * (min + max);
    }

    glm::vec3 Length()
    {
        return max - min;
    }
};

struct Shape
{
    std::vector<Vertex> vertices;
    std::vector<Index>  indices;

    AxisAlignedBox aabb;
};

struct Model
{
    std::vector<Shape> shapes;

    AxisAlignedBox aabb;
};

struct ParserDesc
{
    std::set<std::string> extensions;
};

class Parser
{
  public:
    Parser();
    ~Parser();

    bool Parse(const std::string& file, Model& model);
};