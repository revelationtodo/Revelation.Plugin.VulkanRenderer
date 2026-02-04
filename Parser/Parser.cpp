#include "Parser.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

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

static glm::vec4 GetDiffuseColor(aiMaterial* mat)
{
    aiColor4D diffuseColor;
    if (aiGetMaterialColor(mat, AI_MATKEY_COLOR_DIFFUSE, &diffuseColor) == AI_SUCCESS)
    {
        return glm::vec4(diffuseColor.r, diffuseColor.g, diffuseColor.b, diffuseColor.a);
    }
    return glm::vec4(1.0f);
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

static bool LoadTexture(const std::string& texPath, const aiScene* scene, Texture& tex)
{
    if (texPath[0] == '*') // embedded texture
    {
        int              index = std::atoi(texPath.c_str() + 1);
        const aiTexture* aiTex = scene->mTextures[index];
        if (aiTex->mHeight == 0) // compressed
        {
            stbi_uc* loaded = stbi_load_from_memory(reinterpret_cast<const uint8_t*>(aiTex->pcData), aiTex->mWidth, &tex.width, &tex.height, &tex.channels, STBI_rgb_alpha);
            if (!loaded || tex.width == 0 || tex.height == 0)
            {
                return false;
            }

            tex.channels = 4;
            size_t size  = size_t(tex.width) * size_t(tex.height) * size_t(tex.channels);
            tex.buffer.assign(loaded, loaded + size);
            stbi_image_free(loaded);
        }
        else // uncompressed
        {
            size_t byteSize = aiTex->mWidth * aiTex->mHeight * 4;
            tex.buffer.resize(byteSize);
            memcpy(tex.buffer.data(), aiTex->pcData, byteSize);
            tex.width  = aiTex->mWidth;
            tex.height = aiTex->mHeight;
        }
    }
    else // regular image file
    {
        stbi_uc* loaded = stbi_load(texPath.c_str(), &tex.width, &tex.height, &tex.channels, STBI_rgb_alpha);
        if (!loaded || tex.width == 0 || tex.height == 0)
        {
            return false;
        }

        tex.channels = 4;
        size_t size  = size_t(tex.width) * size_t(tex.height) * size_t(tex.channels);
        tex.buffer.assign(loaded, loaded + size);
        stbi_image_free(loaded);
    }

    tex.id = texPath;
    return true;
}

static glm::mat4 ToGlm(const aiMatrix4x4& m)
{
    // xelement-wise conversion to glm::mat4 (column-major)
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

static void ExpandAabb(AxisAlignedBox& aabb, const glm::vec3& p)
{
    aabb.min = glm::min(aabb.min, p);
    aabb.max = glm::max(aabb.max, p);
}

static void MergeAabb(AxisAlignedBox& dst, const AxisAlignedBox& src)
{
    dst.min = glm::min(dst.min, src.min);
    dst.max = glm::max(dst.max, src.max);
}

static AxisAlignedBox TransformAabb(const AxisAlignedBox& local, const glm::mat4& M)
{
    const glm::vec3 mn = local.min;
    const glm::vec3 mx = local.max;

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

    AxisAlignedBox out;
    for (const auto& c : corners)
    {
        glm::vec4 p = M * glm::vec4(c, 1.0f);
        ExpandAabb(out, glm::vec3(p));
    }
    return out;
}

// ------------------------------------------------------------
// Parse one aiMesh -> Mesh
// Mesh::trans will be set to meshGlobalTrans (global transform)
// Node::aabb will be merged with meshWorldAabb (world-space)
// ------------------------------------------------------------
static bool ParseOneMesh(const aiScene*     scene,
                         const aiMesh*      aimesh,
                         const std::string& assetDir,
                         const glm::mat4&   meshGlobalTrans,
                         Mesh&              outMesh,
                         AxisAlignedBox&    outMeshWorldAabb)
{
    if (!scene || !aimesh)
        return false;

    Mesh mesh;
    mesh.trans = meshGlobalTrans;

    mesh.vertices.reserve(aimesh->mNumVertices);

    const bool hasNormals = aimesh->HasNormals();
    const bool hasUV0     = aimesh->HasTextureCoords(0);

    // vertices + local aabb
    for (unsigned int vi = 0; vi < aimesh->mNumVertices; ++vi)
    {
        Vertex v{};

        const aiVector3D& p = aimesh->mVertices[vi];
        v.pos               = glm::vec3(p.x, p.y, p.z);
        ExpandAabb(mesh.aabb, v.pos);

        if (hasNormals)
        {
            const aiVector3D& n = aimesh->mNormals[vi];
            v.normal            = glm::vec3(n.x, n.y, n.z);
        }
        else
        {
            v.normal = glm::vec3(0.0f, 1.0f, 0.0f);
        }

        if (hasUV0)
        {
            const aiVector3D& t = aimesh->mTextureCoords[0][vi];
            v.uv                = glm::vec2(t.x, t.y);

            // keep your original flip
            v.uv.y = 1.0f - v.uv.y;
        }
        else
        {
            v.uv = glm::vec2(0.0f);
        }

        mesh.vertices.push_back(v);
    }

    // indices
    std::vector<unsigned int> tmp;
    tmp.reserve(aimesh->mNumFaces * 3);

    for (unsigned int fi = 0; fi < aimesh->mNumFaces; ++fi)
    {
        const aiFace& face = aimesh->mFaces[fi];
        if (face.mNumIndices != 3)
            continue;

        tmp.push_back(face.mIndices[0]);
        tmp.push_back(face.mIndices[1]);
        tmp.push_back(face.mIndices[2]);
    }

    unsigned int maxIndex = 0;
    for (unsigned int idx : tmp)
        maxIndex = (idx > maxIndex) ? idx : maxIndex;

    if (maxIndex > std::numeric_limits<Index>::max())
        return false;

    mesh.indices.reserve(tmp.size());
    for (unsigned int idx : tmp)
        mesh.indices.push_back(static_cast<Index>(idx));

    // diffuse texture path
    if (aimesh->mMaterialIndex >= 0 && scene->mMaterials)
    {
        auto* mat = scene->mMaterials[aimesh->mMaterialIndex];

        //////////////////////////////////////////////////////////////////////////
        // diffuse color
        mesh.material.baseDiffuseColor = GetDiffuseColor(mat);

        // diffuse texture
        std::string diffuseTex = GetDiffuseTexRef(mat);
        diffuseTex             = JoinPath(assetDir, diffuseTex);
        LoadTexture(diffuseTex, scene, mesh.material.diffuseTexture);
        //////////////////////////////////////////////////////////////////////////
    }

    // world aabb (transform local aabb by global matrix)
    outMeshWorldAabb = TransformAabb(mesh.aabb, meshGlobalTrans);

    outMesh = std::move(mesh);
    return true;
}

// ------------------------------------------------------------
// Recursively convert aiNode tree -> your Node tree
//
// Node::trans = local transform (aiNode::mTransformation)
// Mesh::trans = global transform (parentGlobal * nodeLocal)
//
// Node::aabb is world-space AABB merged from:
// - transformed mesh aabb under this node
// - children's aabb
// ------------------------------------------------------------
static void ProcessNodeRecursive(const aiScene*     scene,
                                 const aiNode*      ain,
                                 const std::string& assetDir,
                                 const glm::mat4&   parentGlobal,
                                 Node&              outNode)
{
    // reset
    outNode.children.clear();
    outNode.meshes.clear();
    outNode.aabb  = AxisAlignedBox{};
    outNode.trans = glm::mat4(1.0f);

    if (!scene || !ain)
        return;

    // local + global
    const glm::mat4 local  = ToGlm(ain->mTransformation);
    const glm::mat4 global = parentGlobal * local;

    outNode.trans = local;

    // meshes attached to this node
    outNode.meshes.reserve(ain->mNumMeshes);
    for (unsigned int i = 0; i < ain->mNumMeshes; ++i)
    {
        const unsigned int meshIndex = ain->mMeshes[i];
        if (meshIndex >= scene->mNumMeshes)
            continue;

        const aiMesh* aimesh = scene->mMeshes[meshIndex];

        Mesh           mesh;
        AxisAlignedBox meshWorldAabb;
        if (!ParseOneMesh(scene, aimesh, assetDir, global, mesh, meshWorldAabb))
            continue;

        MergeAabb(outNode.aabb, meshWorldAabb);
        outNode.meshes.push_back(std::move(mesh));
    }

    // children
    outNode.children.reserve(ain->mNumChildren);
    for (unsigned int c = 0; c < ain->mNumChildren; ++c)
    {
        Node child;
        ProcessNodeRecursive(scene, ain->mChildren[c], assetDir, global, child);

        MergeAabb(outNode.aabb, child.aabb);
        outNode.children.push_back(std::move(child));
    }
}

Parser::Parser()  = default;
Parser::~Parser() = default;

bool Parser::Parse(const std::string& file, Node& node)
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
    if (!scene || !scene->mRootNode)
        return false;

    std::filesystem::path filePath(file);
    const std::string     assetDir = filePath.parent_path().string();

    ProcessNodeRecursive(scene, scene->mRootNode, assetDir, glm::mat4(1.0f), node);

    return (!node.meshes.empty() || !node.children.empty());
}
