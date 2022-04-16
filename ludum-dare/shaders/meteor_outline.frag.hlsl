
#include "common.h"

struct In {
    float3 normal: TEXCOORD0;
    float2 uv: TEXCOORD1;
    float3 position: TEXCOORD2;
};

struct Out {
    float4 color: SV_Target0;
    float4 opaque_color: SV_Target1;
};

[[vk::binding(2)]] sampler tex_sampler;
[[vk::binding(3)]] Texture2D<float3> texture2;

Out main(In input) {
    Out output;

    output.color = float4(texture2.Sample(tex_sampler, input.uv), 1.0 / 3.0);
    output.color = linear_to_srgb(output.color);
    output.opaque_color = output.color;

    return output;
}
