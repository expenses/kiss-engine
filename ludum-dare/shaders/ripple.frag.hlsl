#include "common.h"

[[vk::binding(2)]] cbuffer _ {
    float3 meteor_position;
};

[[vk::binding(3)]] sampler tex_sampler;
[[vk::binding(4)]] Texture2D<float4> ripple_tex;


struct In {
    [[vk::location(0)]] float2 uv: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
    [[vk::location(2)]] float3 position: TEXCOORD2;
};

struct Out {
    float4 color: SV_TARGET0;
    float4 color2: SV_TARGET1;
};

Out main(In input) {
    Out output;

    float4 color = ripple_tex.Sample(tex_sampler, input.uv);

    color.xyz *= shadow_factor(input.position, meteor_position);
    color.w *= 0.5;

    output.color = color;
    output.color2 = color;

    return output;
}