#include "Parser.h"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <cstdint>
#include <string>
#include <vector>
#include <filesystem>

static std::string GetDiffuseTexRef(aiMaterial* mat)
{
    aiString texPath;

    if (mat->GetTextureCount(aiTextureType_BASE_COLOR) > 0 &&
        mat->GetTexture(aiTextureType_BASE_COLOR, 0, &texPath) == AI_SUCCESS)
    {
        return texPath.C_Str();
    }

    if (mat->GetTextureCount(aiTextureType_DIFFUSE) > 0 &&
        mat->GetTexture(aiTextureType_DIFFUSE, 0, &texPath) == AI_SUCCESS)
    {
        return texPath.C_Str();
    }

    return "";
}

static std::string JoinPath(const std::string& dir, const std::string& rel)
{
    if (rel.empty() || rel[0] == '*')
    {
        return rel;
    }

    std::filesystem::path texPath(rel);
    if (texPath.is_absolute())
    {
        return std::filesystem::weakly_canonical(texPath).string();
    }

    std::filesystem::path base(dir);
    std::filesystem::path full = base / texPath;
    return std::filesystem::weakly_canonical(full).string();
}

Parser::Parser()
{
}

Parser::~Parser()
{
}

bool Parser::Parse(const std::string& file, Model& model)
{

    const unsigned int flags =
        aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcess_GenSmoothNormals |
        aiProcess_CalcTangentSpace |
        aiProcess_ImproveCacheLocality |
        aiProcess_OptimizeMeshes |
        aiProcess_OptimizeGraph;

    Assimp::Importer importer;
    const aiScene*   scene = importer.ReadFile(file, flags);
    if (nullptr == scene || !scene->mRootNode)
    {
        return false;
    }

    model.shapes.clear();
    model.shapes.reserve(scene->mNumMeshes);
    for (unsigned int mi = 0; mi < scene->mNumMeshes; ++mi)
    {
        const aiMesh* mesh = scene->mMeshes[mi];
        if (nullptr == mesh)
        {
            continue;
        }

        Shape shape;
        shape.vertices.reserve(mesh->mNumVertices);

        const bool hasNormals = mesh->HasNormals();
        const bool hasUV0     = mesh->HasTextureCoords(0);

        for (unsigned int vi = 0; vi < mesh->mNumVertices; ++vi)
        {
            Vertex v{};

            const aiVector3D& p = mesh->mVertices[vi];
            v.pos               = glm::vec3(p.x, p.y, p.z);
            shape.aabb.min      = glm::min(shape.aabb.min, v.pos);
            shape.aabb.max      = glm::max(shape.aabb.max, v.pos);

            if (hasNormals)
            {
                const aiVector3D& n = mesh->mNormals[vi];
                v.normal            = glm::vec3(n.x, n.y, n.z);
            }
            else
            {
                v.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            }

            if (hasUV0)
            {
                const aiVector3D& t = mesh->mTextureCoords[0][vi];
                v.uv                = glm::vec2(t.x, t.y);

                v.uv.y = 1.0f - v.uv.y;
            }
            else
            {
                v.uv = glm::vec2(0.0f);
            }

            shape.vertices.push_back(v);
        }

        std::vector<unsigned int> tmp;
        tmp.reserve(mesh->mNumFaces * 3);
        for (unsigned int fi = 0; fi < mesh->mNumFaces; ++fi)
        {
            const aiFace& face = mesh->mFaces[fi];
            if (face.mNumIndices != 3)
            {
                continue;
            }

            tmp.push_back(face.mIndices[0]);
            tmp.push_back(face.mIndices[1]);
            tmp.push_back(face.mIndices[2]);
        }

        unsigned int maxIndex = 0;
        for (unsigned int idx : tmp)
        {
            maxIndex = (idx > maxIndex) ? idx : maxIndex;
        }

        if (maxIndex > std::numeric_limits<Index>::max())
        {
            return false;
        }

        shape.indices.reserve(tmp.size());
        for (unsigned int idx : tmp)
        {
            shape.indices.push_back(static_cast<Index>(idx));
        }

        if (shape.vertices.size() > 0)
        {
            model.aabb.min = glm::min(model.aabb.min, shape.aabb.min);
            model.aabb.max = glm::max(model.aabb.max, shape.aabb.max);
        }

        auto* mat     = scene->mMaterials[mesh->mMaterialIndex];
        shape.diffuse = GetDiffuseTexRef(mat);
        std::filesystem::path filePath(file);
        shape.diffuse = JoinPath(filePath.parent_path().string(), shape.diffuse);

        model.shapes.push_back(std::move(shape));
    }

    return !model.shapes.empty();
}
