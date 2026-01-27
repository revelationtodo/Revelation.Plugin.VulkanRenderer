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

using Index = uint16_t;

struct Shape
{
    std::vector<Vertex> vertices;
    std::vector<Index>  indices;

    // texture
};

struct Model
{
    std::vector<Shape> shapes;
};

struct ParserDesc
{
    std::set<std::string> extensions;
};

class IParser
{
  public:
    virtual bool CanParse(const std::string& file)            = 0;
    virtual bool Parse(const std::string& file, Model& model) = 0;
};