
#include "common.h"

struct In {
    [[vk::location(0)]] float3 normal: TEXCOORD0;
    [[vk::location(1)]] float2 uv: TEXCOORD1;
    [[vk::location(2)]] float3 position: TEXCOORD2;

};

struct Out {
    float4 color: SV_TARGET0;
    float4 color2: SV_TARGET1;
};

[[vk::binding(1)]] cbuffer _ {
    float3 meteor_position;
};

[[vk::binding(2)]] sampler tex_sampler;
[[vk::binding(3)]] Texture2D<float3> forest_tex;

Out main(In input) {
    Out output;

    float diffuse = max(dot(normalize(input.normal), SUN_DIR), 0.1);

    output.color = float4(forest_tex.Sample(tex_sampler, input.uv) * diffuse, 1.0);

    output.color *= shadow_factor(input.position, meteor_position);
    output.color2 = output.color;


    return output;
}
