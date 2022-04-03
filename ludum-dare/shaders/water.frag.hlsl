#include "common.h"

[[vk::binding(0)]] cbuffer _ {
    row_major float4x4 matrices;
    float3 player_position;
    float player_facing;
    float3 camera_position;
    float time;
    float2 window_size;
};

[[vk::binding(1)]] Texture2D<float4> opaque_tex;
[[vk::binding(2)]] sampler tex_sampler;
[[vk::binding(3)]] Texture2D<float4> depth_map_tex;

[[vk::binding(4)]] cbuffer _ {
    float3 meteor_position;
};

struct In {
    [[vk::location(0)]] float2 uv: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
    [[vk::location(2)]] float3 position: TEXCOORD2;
    float4 coord: SV_Position;
};

struct Out {
    float4 color: SV_TARGET0;
};

Out main(In input) {
    Out output;

    float height = depth_map_tex.SampleLevel(tex_sampler, input.uv, 0).r;

    // Compute light attenuation using Beer's law.

    float3 transmitted_light = opaque_tex.Sample(tex_sampler, input.coord.xy / window_size);

    float attenuation_distance = 0.5;
    float transmission_distance = abs(-height);
    float3 attenuation_color = float3(0.25, 0.25, 1.0);

    attenuation_color *= shadow_factor(input.position, meteor_position) * 0.5 + 0.5;

    float3 attenuation_coefficient = -log(attenuation_color) / attenuation_distance;

    float3 transmittance = exp(-attenuation_coefficient * transmission_distance);

    output.color = float4(transmitted_light * transmittance, 1.0);

    return output;
}