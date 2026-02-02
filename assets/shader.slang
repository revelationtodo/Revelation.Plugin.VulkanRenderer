struct VSInput
{
    float3 Pos;
    float3 Normal;
    float2 UV;
};

Sampler2D textures[];

struct ShaderData
{
    float4x4 projection;
    float4x4 view;
    float4x4 model;

    float4 lightPos;
    float4 cameraPos;
};

struct PushConstant
{
    uint64_t shaderDataAddr;
    int textureIndex;
};

struct VSOutput
{
    float4 Pos : SV_POSITION;

    float2 UV;

    // world space data for lighting
    float3 NormalWS;
    float3 LightVecWS;
    float3 ViewVecWS;

    int TextureIndex;
};

[[vk::push_constant]]
PushConstant pc;

[shader("vertex")]
VSOutput main(VSInput input, uint instanceIndex : SV_VulkanInstanceID)
{
    VSOutput output;

    ShaderData* shaderData = (ShaderData*)pc.shaderDataAddr;

    float4x4 M = shaderData->model;
    float4x4 V = shaderData->view;
    float4x4 P = shaderData->projection;

    // World position
    float4 posWS4 = mul(M, float4(input.Pos, 1.0));
    float3 posWS  = posWS4.xyz;

    float3 normalWS = input.Normal;

    output.UV = input.UV;

    // Clip space position
    output.Pos = mul(P, mul(V, posWS4));

    // World-space lighting vectors
    float3 lightPosWS  = shaderData->lightPos.xyz;
    float3 cameraPosWS = shaderData->cameraPos.xyz;

    output.LightVecWS = lightPosWS - posWS;
    output.ViewVecWS  = cameraPosWS - posWS;
    output.NormalWS   = normalWS;

    output.TextureIndex = pc.textureIndex;

    return output;
}

[shader("fragment")]
float4 main(VSOutput input)
{
    // Phong lighting in world space
    float3 N = normalize(input.NormalWS);
    float3 L = normalize(input.LightVecWS);
    float3 V = normalize(input.ViewVecWS);
    float3 R = reflect(-L, N);

    float  ndotl    = max(dot(N, L), 0.0);
    float3 diffuse  = max(ndotl, 0.0025);
    float3 specular = pow(max(dot(R, V), 0.0), 16.0) * 0.75;

    // Sample texture
    float3 albedo = float3(1.0, 1.0, 1.0);
    if (input.TextureIndex >= 0)
    {
        albedo = textures[input.TextureIndex].Sample(input.UV).rgb;
    }

    float3 ambient = float3(0.05, 0.05, 0.05) * albedo;
    return float4(ambient + diffuse * albedo + specular, 1.0);
}
