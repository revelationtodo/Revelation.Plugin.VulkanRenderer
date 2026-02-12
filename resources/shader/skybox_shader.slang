struct VSInput
{
    float3 position : POSITION;
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

struct PushConstant
{
    uint64_t frameUniformsAddr;
    int      meshUniformsIndex;
    int      _pad0;
    int      _pad1;
};

struct VSOutput
{
    float4 positionCS : SV_Position;
    float3 dir        : TEXCOORD0;
};

// set=0 binding=0 : combined image sampler cubemap
[[vk::binding(0, 0)]]
SamplerCube gSkybox;

[[vk::push_constant]]
PushConstant pc;

[shader("vertex")]
VSOutput main(VSInput input)
{
    VSOutput o;
    FrameUniforms* fu = (FrameUniforms*)pc.frameUniformsAddr;

    float4x4 viewNoTrans = fu->view;
    viewNoTrans[0][3] = 0.0;
    viewNoTrans[1][3] = 0.0;
    viewNoTrans[2][3] = 0.0;

    float4 pos = mul(viewNoTrans, float4(input.position, 1.0));
    pos = mul(fu->proj, pos);

    pos.z = pos.w * 0.999999;
    o.positionCS = pos;

    o.dir = input.position;
    return o;
}

float3 ToneMap_ACES(float3 x)
{
    // Narkowicz 2015 (ACES fitted)
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

[shader("fragment")]
float4 main(VSOutput input) : SV_Target
{
    float3 d_new = normalize(input.dir);
    float3 d_old = float3(d_new.x, d_new.z, d_new.y);

    // HDR sample (linear, can be > 1)
    float3 hdr = gSkybox.Sample(d_old).rgb;

    float exposure = 0.1;
    hdr *= exposure;

    float3 ldr = ToneMap_ACES(hdr);
    ldr = pow(ldr, 1.0 / 2.2);

    return float4(ldr, 1.0);
}
