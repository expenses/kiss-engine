#include "constants.h"


struct In {
    [[vk::location(0)]] float2 normal: TEXCOORD0;
};


struct Out {
        [[vk::location(0)]] float4 color: TEXCOORD0;

};

Out main(In input) {
    Out output;

    float diffuse = max(dot(normalize(input.normal), SUN_DIR), 0.0);

    output.color = float4(float3(diffuse), 1.0);

    return output;
}