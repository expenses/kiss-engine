
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

[[vk::binding(1)]] cbuffer _ {
    float3 meteor_position;
};

[[vk::binding(2)]] sampler tex_sampler;
[[vk::binding(3)]] Texture2D<float3> forest_tex;

Out main(In input) {
    Out output;

    float diffuse = max(dot(normalize(input.normal), SUN_DIR), 0.1);

    float3 color = forest_tex.Sample(tex_sampler, input.uv) * diffuse;

    color *= shadow_factor(input.position, meteor_position);

    output.color = float4(color, 1.0);
    output.opaque_color = output.color;


    return output;
}
