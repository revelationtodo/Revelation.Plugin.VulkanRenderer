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

    float4   lightPosWS;
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
};

struct VSOutput
{
    float4 positionCS : SV_POSITION;
    float2 uv         : TEXCOORD0;

    // World-space data for lighting
    float3 normalWS   : TEXCOORD1;
    float3 lightVecWS : TEXCOORD2;
    float3 viewVecWS  : TEXCOORD3;

    int    meshIndex  : TEXCOORD4;
};

// [[vk::binding(binding, set)]]
[[vk::binding(0, 0)]]
Sampler2D gTextures[];

// [[vk::binding(binding, set)]]
[[vk::binding(0, 1)]]
StructuredBuffer<MeshUniforms> gMeshUniforms;

[[vk::push_constant]]
PushConstants gPC;

[shader("vertex")]
VSOutput main(VSInput input, uint instanceIndex : SV_VulkanInstanceID)
{
    VSOutput o;

    FrameUniforms* frame = (FrameUniforms*)gPC.frameUniformsAddr;

    float4x4 M = gMeshUniforms[gPC.meshIndex].model;
    float4x4 V = frame->view;
    float4x4 P = frame->proj;

    float4 posWS4 = mul(M, float4(input.position, 1.0));
    float3 posWS  = posWS4.xyz;

    o.uv = input.uv;

    // Clip-space position
    o.positionCS = mul(P, mul(V, posWS4));

    // World-space lighting vectors
    float3 lightPosWS  = frame->lightPosWS.xyz;
    float3 cameraPosWS = frame->cameraPosWS.xyz;

    o.lightVecWS = lightPosWS - posWS;
    o.viewVecWS  = cameraPosWS - posWS;
    o.normalWS   = input.normal;

    o.meshIndex  = gPC.meshIndex;

    return o;
}

[shader("fragment")]
float4 main(VSOutput input) : SV_Target
{
    float3 N = normalize(input.normalWS);
    float3 L = normalize(input.lightVecWS);
    float3 V = normalize(input.viewVecWS);
    float3 R = reflect(-L, N);

    float  ndotl   = max(dot(N, L), 0.0);
    float3 diffuse = max(ndotl, 0.0025);
    float3 spec    = pow(max(dot(R, V), 0.0), 16.0) * 0.75;

    MeshUniforms mesh = gMeshUniforms[input.meshIndex];

    float3 albedo = mesh.baseColor.rgb;
    if (mesh.diffuseTexIndex >= 0)
    {
        albedo = gTextures[mesh.diffuseTexIndex].Sample(input.uv).rgb;
    }

    float3 ambient = float3(0.05, 0.05, 0.05) * albedo;
    return float4(ambient + diffuse * albedo + spec, 1.0);
}
