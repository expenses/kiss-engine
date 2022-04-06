#include "common.h"

[[vk::binding(1)]] cbuffer _ {
    float3 meteor_position;
};

[[vk::binding(2)]] sampler tex_sampler;
[[vk::binding(3)]] sampler linear_sampler;

[[vk::binding(4)]] Texture2D<float3> ground_textures[];

struct In {
    float2 uv: TEXCOORD0;
    float3 normal: TEXCOORD1;
    float3 position: TEXCOORD2;
};

struct Out {
    float4 color: SV_Target0;
    float4 opaque_color: SV_Target1;
};

Out main(In input) {
    Out output;

    float diffuse = max(dot(normalize(input.normal), SUN_DIR), 0.0);

    float2 uv = input.uv * 10.0;

    uint t_grass = 0;
    uint t_sand = 1;
    uint t_rock = 2;
    uint t_forest = 3;
    uint t_forest_map = 4;

    float3 shore = ground_textures[t_sand].Sample(tex_sampler, uv);
    float3 grass = ground_textures[t_grass].Sample(tex_sampler, uv);
    float3 rock = ground_textures[t_rock].Sample(tex_sampler, uv);
    float3 forest = ground_textures[t_forest].Sample(tex_sampler, uv);

    float forest_map = ground_textures[t_forest_map].Sample(linear_sampler, input.uv).r;

    float3 greenery = lerp(grass, forest, forest_map);

    float3 terrain = lerp(greenery, rock, smoothstep(4.0, 5.0, input.position.y));
    terrain = lerp(shore, terrain, smoothstep(0.1, 0.4, input.position.y));
    terrain = lerp(terrain, 1.0, smoothstep(7.0, 7.5, input.position.y));
    float3 color = terrain * diffuse;

    color *= shadow_factor(input.position, meteor_position);

    output.color = float4(color, 1.0);
    output.opaque_color = float4(color, 1.0);

    return output;
}