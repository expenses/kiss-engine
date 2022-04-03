#include "common.h"

struct In {
    [[vk::location(0)]] float2 uv: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
    [[vk::location(2)]] float3 position: TEXCOORD2;
};

struct Out {
    float4 color: SV_TARGET0;
};


[[vk::binding(1)]] cbuffer _ {
    float3 meteor_position;
};

Out main(In input) {
    Out output;

    float diffuse = max(dot(normalize(input.normal), SUN_DIR), 0.0);

    float3 color = float3(0.5, 0.0, 0.0) * diffuse;

    output.color = float4(color, 1.0);

    output.color *= shadow_factor(input.position, meteor_position);

    return output;
}