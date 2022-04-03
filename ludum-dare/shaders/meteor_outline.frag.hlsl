
#include "common.h"

struct In {
    [[vk::location(0)]] float3 normal: TEXCOORD0;
    [[vk::location(1)]] float2 uv: TEXCOORD1;
    [[vk::location(2)]] float3 position: TEXCOORD2;
};

struct Out {
    float4 color: SV_TARGET0;
};

[[vk::binding(2)]] sampler tex_sampler;
[[vk::binding(3)]] Texture2D<float3> texture2;

Out main(In input) {
    Out output;

    output.color = float4(texture2.Sample(tex_sampler, input.uv), 1.0 / 3.0);

    return output;
}
