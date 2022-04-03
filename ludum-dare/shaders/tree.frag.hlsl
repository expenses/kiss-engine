
#include "constants.h"

struct In {
    [[vk::location(0)]] float3 normal: TEXCOORD0;
    [[vk::location(1)]] float2 uv: TEXCOORD1;
    [[vk::location(2)]] float3 position: TEXCOORD2;

};

struct Out {
    float4 color: SV_TARGET0;
};

[[vk::binding(1)]] cbuffer _ {
    float3 meteor_position;
};

[[vk::binding(2)]] sampler tex_sampler;
[[vk::binding(3)]] Texture2D<float3> forest_tex;


float shadow_factor(float3 position, float3 meteor_position) {
    float2 pos_2d = float2(position.x, position.z);
    float2 meteor_pos_2d = float2(meteor_position.x, meteor_position.z);

    float shadow_scale = (150.0 + position.y - meteor_position.y) * 0.01;

    float ambient = 0.025;

    return max(smoothstep(shadow_scale * 0.9, shadow_scale * 1.1, distance(pos_2d, meteor_pos_2d)), ambient);
}

Out main(In input) {
    Out output;

    float diffuse = max(dot(normalize(input.normal), SUN_DIR), 0.0);

    output.color = float4(forest_tex.Sample(tex_sampler, input.uv) * diffuse, 1.0);

    output.color *= shadow_factor(input.position, meteor_position);


    return output;
}
