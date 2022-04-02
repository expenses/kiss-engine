#include "constants.h"

[[vk::binding(1)]] cbuffer _ {
    float3 meteor_position;
};


struct In {
    [[vk::location(0)]] float2 uv: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
    [[vk::location(2)]] float3 position: TEXCOORD2;
};



struct Out {
    float4 color: SV_TARGET0;
};

Out main(In input) {
    Out output;


    float diffuse = max(dot(normalize(input.normal), SUN_DIR), 0.0);

    float3 shore = float3(0.5, 0.5, 0.0);
    float3 grass = float3(0.0, 0.5, 0.0);
    float3 rock = float3(0.25);

    float3 terrain = lerp(grass, rock, smoothstep(4.0, 5.0, input.position.y));
    terrain = lerp(shore, terrain, smoothstep(0.1, 0.4, input.position.y));
    terrain = lerp(terrain, float3(1.0), smoothstep(7.0, 7.5, input.position.y));
    float3 color = terrain * diffuse;


    float2 pos_2d = float2(input.position.x, input.position.z);
    float2 meteor_pos_2d = float2(meteor_position.x, meteor_position.z);

    float shadow_scale = (150.0 - meteor_position.y) * 0.01;

    float ambient = 0.025;

    color *= max(smoothstep(shadow_scale - 0.1, shadow_scale + 0.1, distance(pos_2d, meteor_pos_2d)), ambient);

    output.color = float4(color, 1.0);

    return output;
}