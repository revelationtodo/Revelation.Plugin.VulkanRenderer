struct VSInput
{
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 uv       : TEXCOORD0;
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

    float4   baseColor;
    int      diffuseTexIndex;
    int      _pad0;
    int      _pad1;
    int      _pad2;
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

    // World-space data for lighting
    float3 normalWS   : TEXCOORD1;
    float3 lightVecWS : TEXCOORD2; // for directional light: constant per-vertex
    float3 viewVecWS  : TEXCOORD3;

    int    meshIndex  : TEXCOORD4;
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

    // World position
    float4 posWS4 = mul(M, float4(input.position, 1.0));
    float3 posWS  = posWS4.xyz;

    o.uv = input.uv;

    // Clip-space
    o.positionCS = mul(P, mul(V, posWS4));

    // Camera vector (world-space)
    float3 cameraPosWS = frame->cameraPosWS.xyz;
    o.viewVecWS = cameraPosWS - posWS;

    // Directional light:
    // frame->lightDirWS.xyz is the direction the light travels (from light to scene)
    // For shading we want L = direction from surface toward the light => negate.
    float3 lightDirWS = safeNormalize(frame->lightDirWS.xyz);
    o.lightVecWS = -lightDirWS;

    // Normal to world-space
    // NOTE: If you have non-uniform scaling, use inverse-transpose normal matrix instead.
    float3x3 normalMat = (float3x3)M;
    o.normalWS = safeNormalize(mul(normalMat, input.normal));

    o.meshIndex = gPC.meshIndex;
    return o;
}

[shader("fragment")]
float4 main(VSOutput input) : SV_Target
{
    MeshUniforms mesh = gMeshUniforms[input.meshIndex];

    float3 albedo = mesh.baseColor.rgb;
    if (mesh.diffuseTexIndex >= 0)
    {
        albedo = gTextures[mesh.diffuseTexIndex].Sample(input.uv).rgb;
    }

    float3 N = safeNormalize(input.normalWS);
    float3 L = safeNormalize(input.lightVecWS);
    float3 V = safeNormalize(input.viewVecWS);

    // Basic Blinn/Phong-ish spec
    float ndotl = max(dot(N, L), 0.0);

    // Tiny floor to avoid fully black when nearly grazing, keep your original intent
    float3 diffuse = max(ndotl, 0.0025) * albedo;

    float3 R = reflect(-L, N);
    float  specPow = 16.0;
    float3 spec    = pow(max(dot(R, V), 0.0), specPow) * 0.75;

    float3 ambient = float3(0.05, 0.05, 0.05) * albedo;

    return float4(ambient + diffuse + spec, 1.0);
}
