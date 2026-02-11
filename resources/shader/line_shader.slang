struct VSInput
{
    float3 position : POSITION;
    float4 color    : COLOR0;
};

struct FrameUniforms
{
    float4x4 proj;
    float4x4 view;
    float4   lightDirWS;
    float4   cameraPosWS;
};

struct PushConstant
{
    uint64_t frameUniformsAddr;
    int      meshIndex;
    int      _pad0;
    int      _pad1;
};

[[vk::push_constant]]
PushConstant pc;

struct VSOutput
{
    float4 positionCS : SV_Position;
    float4 color      : COLOR0;
};

[shader("vertex")]
VSOutput main(VSInput i)
{
    VSOutput o;

    FrameUniforms* fu = (FrameUniforms*)pc.frameUniformsAddr;

    float4 posWS = float4(i.position, 1.0);
    float4 posVS = mul(fu->view, posWS);
    o.positionCS = mul(fu->proj, posVS);

    o.color = i.color;
    return o;
}

[shader("fragment")]
float4 main(VSOutput i) : SV_Target
{
    return i.color;
}
