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


    float alpha = 0.5;

    output.color = float4(float3(0.5, 0.0, 0.5) * alpha, alpha);

    return output;
}