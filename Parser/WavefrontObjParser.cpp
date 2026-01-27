#include "WavefrontObjParser.h"
#include <filesystem>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

WavefrontObjParser::WavefrontObjParser()
{
}

WavefrontObjParser::~WavefrontObjParser()
{
}

bool WavefrontObjParser::CanParse(const std::string& file)
{
    std::filesystem::path filePath = std::filesystem::path(file);
    return filePath.extension().string() == ".obj";
}

bool WavefrontObjParser::Parse(const std::string& file, Model& model)
{
    tinyobj::attrib_t                attrib;
    std::vector<tinyobj::shape_t>    shapes;
    std::vector<tinyobj::material_t> materials;
    std::string                      err;
    bool                             ok = tinyobj::LoadObj(&attrib, &shapes, &materials, nullptr, &err, file.c_str());
    if (!err.empty() || !ok)
    {
        return false;
    }

    for (const auto& shape : shapes)
    {
        Shape s;
        s.vertices.reserve(shape.mesh.indices.size());
        s.indices.reserve(shape.mesh.indices.size());

        for (const auto& idx : shape.mesh.indices)
        {
            Vertex v{};

            v.pos = {
                attrib.vertices[3 * idx.vertex_index + 0],
                attrib.vertices[3 * idx.vertex_index + 1],
                attrib.vertices[3 * idx.vertex_index + 2],
            };

            if (idx.normal_index >= 0 && !attrib.normals.empty())
            {
                v.normal = {
                    attrib.normals[3 * idx.normal_index + 0],
                    attrib.normals[3 * idx.normal_index + 1],
                    attrib.normals[3 * idx.normal_index + 2],
                };
            }
            else
            {
                v.normal = {0, 0, 1};
            }

            if (idx.texcoord_index >= 0 && !attrib.texcoords.empty())
            {
                v.uv = {
                    attrib.texcoords[2 * idx.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * idx.texcoord_index + 1],
                };
            }
            else
            {
                v.uv = {0, 0};
            }

            s.vertices.push_back(v);
            s.indices.push_back(static_cast<uint32_t>(s.indices.size()));
        }

        model.shapes.emplace_back(std::move(s));
    }

    return true;
}
