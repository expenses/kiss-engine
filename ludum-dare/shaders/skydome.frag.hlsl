#include "common.h"

[[vk::binding(1)]] sampler tex_sampler;
[[vk::binding(2)]] Texture2D<float4> skydome_tex;

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

    float4 color = skydome_tex.Sample(tex_sampler, input.uv);

    output.color = color;
    output.color2 = color;

    return output;
}