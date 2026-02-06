#include "Parser.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/norm.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <cstring>

Parser::Parser()  = default;
Parser::~Parser() = default;

bool Parser::Parse(const std::string& file, Node& node)
{
    Assimp::Importer importer;
    const aiScene*   scene = importer.ReadFile(file, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices |
                                                         aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace |
                                                         aiProcess_OptimizeMeshes | aiProcess_OptimizeGraph);

    if (!scene || !scene->mRootNode)
    {
        return false;
    }

    std::string assetDir = std::filesystem::path(file).parent_path().string();
    ProcessNodeRecursive(scene, scene->mRootNode, assetDir, glm::mat4(1.0f), node);
    return true;
}

// ------------------------------------------------------------
// node / mesh parsing
// ------------------------------------------------------------
void Parser::ProcessNodeRecursive(const aiScene* scene, const aiNode* ain, const std::string& assetDir,
                                  const glm::mat4& parentGlobal, Node& outNode) const
{
    outNode = Node();

    glm::mat4 local  = ToGlm(ain->mTransformation);
    glm::mat4 global = parentGlobal * local;
    outNode.trans    = local;

    for (unsigned i = 0; i < ain->mNumMeshes; ++i)
    {
        Mesh           m;
        AxisAlignedBox waabb;
        ParseOneMesh(scene, scene->mMeshes[ain->mMeshes[i]], assetDir, global, m, waabb);
        MergeAabb(outNode.aabb, waabb);
        outNode.meshes.push_back(std::move(m));
    }

    for (unsigned i = 0; i < ain->mNumChildren; ++i)
    {
        Node child;
        ProcessNodeRecursive(scene, ain->mChildren[i], assetDir, global, child);
        MergeAabb(outNode.aabb, child.aabb);
        outNode.children.push_back(std::move(child));
    }
}

bool Parser::ParseOneMesh(const aiScene* scene, const aiMesh* aimesh, const std::string& assetDir,
                          const glm::mat4& meshGlobalTrans, Mesh& outMesh, AxisAlignedBox& outMeshWorldAabb) const
{
    Mesh mesh;
    mesh.trans = meshGlobalTrans;

    for (unsigned i = 0; i < aimesh->mNumVertices; ++i)
    {
        Vertex v;
        auto&  p = aimesh->mVertices[i];

        // position
        v.pos = {p.x, p.y, p.z};

        // normal
        if (aimesh->HasNormals())
        {
            auto& n  = aimesh->mNormals[i];
            v.normal = {n.x, n.y, n.z};
        }

        // uv
        if (aimesh->HasTextureCoords(0))
        {
            auto& t = aimesh->mTextureCoords[0][i];
            v.uv    = {t.x, 1.0f - t.y};
        }

        // vertex color
        if (aimesh->HasVertexColors(0))
        {
            const aiColor4D c = aimesh->mColors[0][i];
            v.color           = {c.r, c.g, c.b, c.a};
        }

        // tangent
        if (aimesh->HasTangentsAndBitangents())
        {
            const aiVector3D& t = aimesh->mTangents[i];
            const aiVector3D& b = aimesh->mBitangents[i];

            glm::vec3 T(t.x, t.y, t.z);
            glm::vec3 B(b.x, b.y, b.z);

            glm::vec3 N = v.normal;
            if (!aimesh->HasNormals() || glm::length2(N) < 1e-20f)
            {
                N = glm::vec3(0.0f, 0.0f, 1.0f);
            }
            else
            {
                N = glm::normalize(N);
            }

            if (glm::length2(T) < 1e-20f)
            {
                T = glm::vec3(1.0f, 0.0f, 0.0f);
            }
            else
            {
                T = glm::normalize(T);
                T = glm::normalize(T - N * glm::dot(N, T)); // Gram-Schmidt
            }

            float handedness = 1.0f;
            if (glm::length2(B) >= 1e-20f)
            {
                B          = glm::normalize(B);
                handedness = (glm::dot(glm::cross(N, T), B) < 0.0f) ? -1.0f : 1.0f;
            }

            v.tangent = glm::vec4(T, handedness);
        }

        // aabb
        ExpandAabb(mesh.aabb, v.pos);

        mesh.vertices.push_back(v);
    }

    for (unsigned i = 0; i < aimesh->mNumFaces; ++i)
    {
        const aiFace& f = aimesh->mFaces[i];
        if (f.mNumIndices == 3)
        {
            mesh.indices.push_back(f.mIndices[0]);
            mesh.indices.push_back(f.mIndices[1]);
            mesh.indices.push_back(f.mIndices[2]);
        }
    }

    if (aimesh->mMaterialIndex >= 0)
    {
        aiMaterial* mat = scene->mMaterials[aimesh->mMaterialIndex];

        // diffuse
        mesh.material.diffuseColor = GetColorByType(mat, TextureType::Diffuse);
        {
            std::string ref = GetTextureRefByType(mat, TextureType::Diffuse);
            std::string tex = JoinPath(assetDir, ref);
            LoadTexture(tex, scene, mesh.material.diffuseTexture);
        }

        // emissive
        mesh.material.emissiveColor = GetColorByType(mat, TextureType::Emissive);
        {
            std::string ref = GetTextureRefByType(mat, TextureType::Emissive);
            std::string tex = JoinPath(assetDir, ref);
            LoadTexture(tex, scene, mesh.material.emissiveTexture);
        }

        // normal
        {
            std::string ref = GetTextureRefByType(mat, TextureType::Normal);
            std::string tex = JoinPath(assetDir, ref);
            LoadTexture(tex, scene, mesh.material.normalTexture);
        }
    }

    outMeshWorldAabb = TransformAabb(mesh.aabb, meshGlobalTrans);
    outMesh          = std::move(mesh);
    return true;
}

std::string Parser::GetTextureRefByType(aiMaterial* mat, TextureType type) const
{
    aiString texPath;

    switch (type)
    {
        case TextureType::Diffuse:
        {
            // glTF / modern formats
            if (mat->GetTextureCount(aiTextureType_BASE_COLOR) > 0 &&
                mat->GetTexture(aiTextureType_BASE_COLOR, 0, &texPath) == AI_SUCCESS)
            {
                return texPath.C_Str();
            }

            // old formats
            if (mat->GetTextureCount(aiTextureType_DIFFUSE) > 0 &&
                mat->GetTexture(aiTextureType_DIFFUSE, 0, &texPath) == AI_SUCCESS)
            {
                return texPath.C_Str();
            }
            return {};
        }
        case TextureType::Emissive:
        {
            if (mat->GetTextureCount(aiTextureType_EMISSIVE) > 0 &&
                mat->GetTexture(aiTextureType_EMISSIVE, 0, &texPath) == AI_SUCCESS)
            {
                return texPath.C_Str();
            }
            return {};
        }
        case TextureType::Normal:
        {
            // glTF / modern formats
            if (mat->GetTextureCount(aiTextureType_NORMALS) > 0 &&
                mat->GetTexture(aiTextureType_NORMALS, 0, &texPath) == AI_SUCCESS)
            {
                return texPath.C_Str();
            }

            // old formats
            if (mat->GetTextureCount(aiTextureType_HEIGHT) > 0 &&
                mat->GetTexture(aiTextureType_HEIGHT, 0, &texPath) == AI_SUCCESS)
            {
                return texPath.C_Str();
            }
            return {};
        }
    }

    return {};
}

glm::vec4 Parser::GetColorByType(aiMaterial* mat, TextureType type) const
{
    aiColor4D c;

    switch (type)
    {
        case TextureType::Diffuse:
        {
            if (aiGetMaterialColor(mat, AI_MATKEY_COLOR_DIFFUSE, &c) == AI_SUCCESS)
            {
                return {c.r, c.g, c.b, c.a};
            }
            return glm::vec4(1.0f);
        }

        case TextureType::Emissive:
        {
            if (aiGetMaterialColor(mat, AI_MATKEY_COLOR_EMISSIVE, &c) == AI_SUCCESS)
            {
                return {c.r, c.g, c.b, c.a};
            }
            return glm::vec4(0.0f);
        }
    }

    return glm::vec4(0.0f);
}

std::string Parser::JoinPath(const std::string& dir, const std::string& rel) const
{
    if (rel.empty() || rel[0] == '*')
    {
        return rel;
    }

    std::filesystem::path p(rel);
    if (p.is_absolute())
    {
        return std::filesystem::weakly_canonical(p).string();
    }

    return std::filesystem::weakly_canonical(std::filesystem::path(dir) / p).string();
}

bool Parser::LoadTexture(const std::string& texPath, const aiScene* scene, Texture& tex) const
{
    if (texPath.empty())
    {
        return false;
    }

    if (texPath[0] == '*')
    {
        int              index = std::atoi(texPath.c_str() + 1);
        const aiTexture* aiTex = scene->mTextures[index];

        if (aiTex->mHeight == 0)
        {
            stbi_uc* loaded = stbi_load_from_memory(reinterpret_cast<const uint8_t*>(aiTex->pcData), aiTex->mWidth,
                                                    &tex.width, &tex.height, &tex.channels, STBI_rgb_alpha);
            if (!loaded)
            {
                return false;
            }

            tex.channels = 4;
            size_t size  = size_t(tex.width) * tex.height * tex.channels;
            tex.buffer.assign(loaded, loaded + size);
            stbi_image_free(loaded);
        }
        else
        {
            tex.width    = aiTex->mWidth;
            tex.height   = aiTex->mHeight;
            tex.channels = 4;

            size_t size = size_t(tex.width) * tex.height * tex.channels;
            tex.buffer.resize(size);
            std::memcpy(tex.buffer.data(), aiTex->pcData, size);
        }
    }
    else
    {
        stbi_uc* loaded = stbi_load(texPath.c_str(), &tex.width, &tex.height, &tex.channels, STBI_rgb_alpha);
        if (!loaded)
        {
            return false;
        }

        tex.channels = 4;
        size_t size  = size_t(tex.width) * tex.height * tex.channels;
        tex.buffer.assign(loaded, loaded + size);
        stbi_image_free(loaded);
    }

    tex.id = texPath;
    return true;
}

// ------------------------------------------------------------
// math helpers
// ------------------------------------------------------------
glm::mat4 Parser::ToGlm(const aiMatrix4x4& m) const
{
    glm::mat4 r;
    r[0][0] = m.a1;
    r[1][0] = m.a2;
    r[2][0] = m.a3;
    r[3][0] = m.a4;
    r[0][1] = m.b1;
    r[1][1] = m.b2;
    r[2][1] = m.b3;
    r[3][1] = m.b4;
    r[0][2] = m.c1;
    r[1][2] = m.c2;
    r[2][2] = m.c3;
    r[3][2] = m.c4;
    r[0][3] = m.d1;
    r[1][3] = m.d2;
    r[2][3] = m.d3;
    r[3][3] = m.d4;
    return r;
}

void Parser::ExpandAabb(AxisAlignedBox& aabb, const glm::vec3& p) const
{
    aabb.min = glm::min(aabb.min, p);
    aabb.max = glm::max(aabb.max, p);
}

void Parser::MergeAabb(AxisAlignedBox& dst, const AxisAlignedBox& src) const
{
    dst.min = glm::min(dst.min, src.min);
    dst.max = glm::max(dst.max, src.max);
}

AxisAlignedBox Parser::TransformAabb(const AxisAlignedBox& local,
                                     const glm::mat4&      M) const
{
    AxisAlignedBox   out;
    const glm::vec3& mn = local.min;
    const glm::vec3& mx = local.max;

    glm::vec3 corners[8] = {
        {mn.x, mn.y, mn.z},
        {mx.x, mn.y, mn.z},
        {mn.x, mx.y, mn.z},
        {mx.x, mx.y, mn.z},
        {mn.x, mn.y, mx.z},
        {mx.x, mn.y, mx.z},
        {mn.x, mx.y, mx.z},
        {mx.x, mx.y, mx.z},
    };

    for (const auto& c : corners)
    {
        glm::vec4 p = M * glm::vec4(c, 1.0f);
        ExpandAabb(out, glm::vec3(p));
    }
    return out;
}
