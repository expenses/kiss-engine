#include "common.h"


struct In {
    [[vk::location(0)]] float3 normal: TEXCOORD0;
    [[vk::location(1)]] float3 dir_to_camera: TEXCOORD1;
};

struct Out {
    [[vk::location(0)]] float4 color: TEXCOORD0;
};

[[vk::binding(1)]] cbuffer _ {
    float3 position;
    float scale;
};

Out main(In input) {
    Out output;

    float fresnel = max(
        dot(normalize(input.normal), normalize(input.dir_to_camera)),
        0.0
    );

    float alpha_scale_multiplier = smoothstep(0.75, 1.25, scale);

    float alpha = (0.75 * (1.0 - fresnel * 0.75)) * alpha_scale_multiplier;

    output.color = float4(float3(0.5, 0.0, 0.5) * alpha, alpha);

    return output;
}