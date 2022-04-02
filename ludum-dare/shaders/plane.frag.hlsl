#include "constants.h"


[[vk::binding(1)]] Texture2D<float4> depth_map_tex;
[[vk::binding(2)]] sampler tex_sampler;


struct In {
    [[vk::location(0)]] float2 uv: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
};



struct Out {
    float4 color: SV_TARGET0;
};

Out main(In input) {
    Out output;

    float height = depth_map_tex.SampleLevel(tex_sampler, input.uv, 0).r;

    float diffuse = max(dot(normalize(input.normal), SUN_DIR), 0.0);

    float3 color = float3(fmod(input.uv, float2(1.0)), 0.0) * diffuse;

    output.color = float4(color, 1.0);

    return output;
}