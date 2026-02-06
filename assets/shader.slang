struct VSInput
{
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 uv       : TEXCOORD0;
    float4 color    : COLOR0;
};

struct FrameUniforms
{
    float4x4 proj;
    float4x4 view;

    // Directional light:
    // xyz = light ray direction in world space (normalized recommended)
    // i.e. direction that light travels (from light toward the scene)
    float4   lightDirWS;

    float4   cameraPosWS;
};

struct MeshUniforms
{
    float4x4 model;

    float4 diffuseColor;
    float4 emissiveColor;

    int4 textureIndexes; // [diffuse, emissive, placeholder, placeholder]
};

struct PushConstants
{
    uint64_t frameUniformsAddr;
    int      meshIndex;
    int      _pad0;
    int      _pad1;
};

struct VSOutput
{
    float4 positionCS : SV_POSITION;
    float2 uv         : TEXCOORD0;

    float3 normalWS   : TEXCOORD1;
    float3 lightVecWS : TEXCOORD2;
    float3 viewVecWS  : TEXCOORD3;

    float4 color      : TEXCOORD4;
    int    meshIndex  : TEXCOORD5;
};

[[vk::binding(0, 0)]]
Sampler2D gTextures[];

[[vk::binding(0, 1)]]
StructuredBuffer<MeshUniforms> gMeshUniforms;

[[vk::push_constant]]
PushConstants gPC;

// Helper: safe normalize (avoid NaNs)
float3 safeNormalize(float3 v)
{
    float len2 = dot(v, v);
    if (len2 < 1e-12) return float3(0.0, 0.0, 1.0);
    return v * rsqrt(len2);
}

[shader("vertex")]
VSOutput main(VSInput input, uint instanceIndex : SV_VulkanInstanceID)
{
    VSOutput o;

    FrameUniforms* frame = (FrameUniforms*)gPC.frameUniformsAddr;
    MeshUniforms meshU = gMeshUniforms[gPC.meshIndex];

    float4x4 M = meshU.model;
    float4x4 V = frame->view;
    float4x4 P = frame->proj;

    float4 posWS4 = mul(M, float4(input.position, 1.0));
    float3 posWS  = posWS4.xyz;

    o.uv    = input.uv;
    o.color = input.color;

    o.positionCS = mul(P, mul(V, posWS4));

    float3 cameraPosWS = frame->cameraPosWS.xyz;
    o.viewVecWS = cameraPosWS - posWS;

    float3 lightDirWS = safeNormalize(frame->lightDirWS.xyz);
    o.lightVecWS = -lightDirWS;

    float3x3 normalMat = (float3x3)M;
    o.normalWS = safeNormalize(mul(normalMat, input.normal));

    o.meshIndex = gPC.meshIndex;
    return o;
}

[shader("fragment")]
float4 main(VSOutput input) : SV_Target
{
    MeshUniforms mesh = gMeshUniforms[input.meshIndex];

    // -------------------------
    // Diffuse (albedo)
    // -------------------------
    float3 albedo = mesh.diffuseColor.rgb;
    albedo *= input.color.rgb; // glTF: baseColorFactor * vertexColor * baseColorTexture
    if (mesh.textureIndexes.x >= 0)
    {
        albedo *= gTextures[mesh.textureIndexes.x].Sample(input.uv).rgb;
    }

    // -------------------------
    // Emissive
    // -------------------------
    float3 emissive = mesh.emissiveColor.rgb;
    if (mesh.textureIndexes.y >= 0)
    {
        // emissiveTexture * emissiveFactor (common convention)
        emissive *= gTextures[mesh.textureIndexes.y].Sample(input.uv).rgb;
    }

    float3 N = safeNormalize(input.normalWS);
    float3 L = safeNormalize(input.lightVecWS);
    float3 V = safeNormalize(input.viewVecWS);

    float ndotl = max(dot(N, L), 0.0);

    float3 diffuse = max(ndotl, 0.0025) * albedo;

    float3 R = reflect(-L, N);
    float  specPow = 16.0;
    float3 spec    = pow(max(dot(R, V), 0.0), specPow) * 0.75;

    float3 ambient = float3(0.05, 0.05, 0.05) * albedo;

    // Emissive added at the end (unlit contribution)
    return float4(ambient + diffuse + spec + emissive, 1.0);
}
