#include "common.h"

[[vk::binding(1)]] cbuffer _ {
    float3 meteor_position;
};

[[vk::binding(2)]] sampler tex_sampler;
[[vk::binding(3)]] sampler linear_sampler;
[[vk::binding(4)]] Texture2D<float3> grass_tex;
[[vk::binding(5)]] Texture2D<float3> sand_tex;
[[vk::binding(6)]] Texture2D<float3> rock_tex;
[[vk::binding(7)]] Texture2D<float3> forest_tex;
[[vk::binding(8)]] Texture2D<float> forest_map_tex;

struct In {
    float2 uv;
    float3 normal;
    float3 position;
};

struct Out {
    float4 color;
    float4 opaque_color;
};

Out main(In input) {
    Out output;

    float diffuse = max(dot(normalize(input.normal), SUN_DIR), 0.0);

    float2 uv = input.uv * 10.0;

    float3 shore = sand_tex.Sample(tex_sampler, uv);
    float3 grass = grass_tex.Sample(tex_sampler, uv);
    float3 rock = rock_tex.Sample(tex_sampler, uv);
    float3 forest = forest_tex.Sample(tex_sampler, uv);

    float forest_map = forest_map_tex.Sample(linear_sampler, input.uv);

    float3 greenery = lerp(grass, forest, forest_map);

    float3 terrain = lerp(greenery, rock, smoothstep(4.0, 5.0, input.position.y));
    terrain = lerp(shore, terrain, smoothstep(0.1, 0.4, input.position.y));
    terrain = lerp(terrain, float3(1.0), smoothstep(7.0, 7.5, input.position.y));
    float3 color = terrain * diffuse;

    color *= shadow_factor(input.position, meteor_position);

    output.color = float4(color, 1.0);
    output.opaque_color = float4(color, 1.0);

    return output;
}