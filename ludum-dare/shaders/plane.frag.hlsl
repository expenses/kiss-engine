#include "constants.h"

struct In {
    [[vk::location(0)]] float2 uv: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
};

struct Out {
    float4 color: SV_TARGET0;
};

Out main(In input) {
    Out output;

    float diffuse = max(dot(normalize(input.normal), SUN_DIR), 0.0);

    float3 color = float3(fmod(input.uv, float2(1.0)), 0.0) * diffuse;

    output.color = float4(float3(color), 1.0);

    return output;
}