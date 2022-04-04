
#include "common.h"

struct In {
    float3 normal;
    float2 uv;
    float3 position;
};

struct Out {
    float4 color;
    float4 opaque_color;
};

[[vk::binding(2)]] sampler tex_sampler;
[[vk::binding(3)]] Texture2D<float3> texture2;

Out main(In input) {
    Out output;

    output.color = float4(texture2.Sample(tex_sampler, input.uv), 1.0 / 3.0);
    output.opaque_color = output.color;

    return output;
}
