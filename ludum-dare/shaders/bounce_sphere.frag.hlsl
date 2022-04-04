#include "common.h"

struct In {
    float3 normal: TEXCOORD0;
    float3 dir_to_camera: TEXCOORD1;
};

struct Out {
    float4 color: SV_Target0;
    float4 opaque_color: SV_Target1;
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
    output.opaque_color = output.color;

    return output;
}