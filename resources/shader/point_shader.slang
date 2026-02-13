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

struct PointsUniforms
{
    float4x4 model;
};

struct PushConstant
{
    uint64_t frameUniformsAddr;
    int64_t  meshUniformsIndex;
    int64_t  pointsUniformsIndex;
};

[[vk::binding(0, 0)]]
StructuredBuffer<PointsUniforms> gPoints;

[[vk::push_constant]]
PushConstant pc;

struct VSOutput
{
    float4 positionCS : SV_Position;
    float4 color      : COLOR0;
    float  pointSize  : SV_PointSize;
};

[shader("vertex")]
VSOutput main(VSInput i)
{
    VSOutput o;

    FrameUniforms* fu = (FrameUniforms*)pc.frameUniformsAddr;

    PointsUniforms pu = gPoints[pc.pointsUniformsIndex];

    float4 posWS = mul(pu.model, float4(i.position, 1.0));
    float4 posVS = mul(fu->view, posWS);
    o.positionCS = mul(fu->proj, posVS);

    o.color     = i.color;
    o.pointSize = 2.0;
    return o;
}

[shader("fragment")]
float4 main(VSOutput i) : SV_Target
{
    return i.color;
}
