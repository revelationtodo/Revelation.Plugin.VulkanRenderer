struct VSInput
{
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 uv       : TEXCOORD0;
    float4 color    : COLOR0;
    float4 tangent  : TANGENT;   // xyz = tangent, w = handedness
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

    int4 textureIndexes; // [diffuse, emissive, normal, orm]
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

    float3 posWS      : TEXCOORD1; // NEW: for per-pixel view vector
    float3 normalWS   : TEXCOORD2;
    float3 tangentWS  : TEXCOORD3; // NEW
    float  tangentW   : TEXCOORD4; // NEW (handedness)

    float4 color      : TEXCOORD5;
    int    meshIndex  : TEXCOORD6;
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

    o.posWS = posWS;
    o.positionCS = mul(P, mul(V, posWS4));

    float3 lightDirWS = safeNormalize(frame->lightDirWS.xyz);

    float3x3 normalMat = (float3x3)M;

    float3 N = safeNormalize(mul(normalMat, input.normal));
    float3 T = safeNormalize(mul(normalMat, input.tangent.xyz));

    // Orthonormalize T against N (helps if import isn't perfect)
    T = safeNormalize(T - N * dot(N, T));

    o.normalWS  = N;
    o.tangentWS = T;
    o.tangentW  = input.tangent.w;

    o.meshIndex = gPC.meshIndex;
    return o;
}

// Decode tangent-space normal from texture (assumes stored in [0,1])
float3 decodeNormalTS(float3 n)
{
    return n * 2.0 - 1.0;
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
        emissive *= gTextures[mesh.textureIndexes.y].Sample(input.uv).rgb;
    }

    // -------------------------
    // Build TBN + Normal Map
    // -------------------------
    float3 N = safeNormalize(input.normalWS);
    float3 T = safeNormalize(input.tangentWS);
    float3 B = safeNormalize(cross(N, T)) * (input.tangentW >= 0.0 ? 1.0 : -1.0);

    float3x3 TBN = float3x3(T, B, N); // columns: T,B,N (matches mul(TBN, nTS))

    if (mesh.textureIndexes.z >= 0)
    {
        float3 nTex = gTextures[mesh.textureIndexes.z].Sample(input.uv).xyz;
        float3 nTS  = decodeNormalTS(nTex);
        // If your normal maps look "inside-out", try flipping Y:
        // nTS.y = -nTS.y;
        N = safeNormalize(mul(TBN, nTS));
    }

    // -------------------------
    // Lighting vectors
    // -------------------------
    FrameUniforms* frame = (FrameUniforms*)gPC.frameUniformsAddr;
    float3 V = safeNormalize(frame->cameraPosWS.xyz - input.posWS);
    float3 L = safeNormalize(frame->lightDirWS.xyz);

    float ndotl = max(dot(N, L), 0.0);

    float3 diffuse = max(ndotl, 0.0025) * albedo;

    float3 R = reflect(-L, N);
    float  specPow = 16.0;
    float3 spec    = pow(max(dot(R, V), 0.0), specPow) * 0.75;

    // -------------------------
    // ORM (AO/Roughness/Metallic)
    // -------------------------
    float ao = 1.0;
    // roughness/metallic
    // float roughness = 1.0;
    // float metallic  = 0.0;

    if (mesh.textureIndexes.w >= 0)
    {
        float3 orm = gTextures[mesh.textureIndexes.w].Sample(input.uv).rgb;
        ao = orm.r;
        // roughness = orm.g;
        // metallic  = orm.b;
    }

    float aoStrength = 1.0; // 0~1
    ao = lerp(1.0, ao, aoStrength);

    // -------------------------
    // Ambient with AO
    // -------------------------
    float3 ambient = float3(0.15, 0.15, 0.15) * albedo * ao;

    return float4(ambient + diffuse + spec + emissive, 1.0);
}
