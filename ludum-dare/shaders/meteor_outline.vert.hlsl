#include "common.h"

[[vk::binding(0)]] cbuffer _ {
    row_major float4x4 matrices;
    float3 player_position;
    float player_facing;
    float3 camera_position;
    float time;
};

[[vk::binding(1)]] cbuffer _ {
    float3 position;
};

struct In {
    [[vk::location(0)]] float3 pos: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
    [[vk::location(2)]] float3 uv: TEXCOORD2;
};

struct Out {
    float4 position: SV_Position;
    [[vk::location(0)]] float3 normal: TEXCOORD0;
    [[vk::location(1)]] float2 uv: TEXCOORD1;
    [[vk::location(2)]] float3 position2: TEXCOORD2;

};

Out main(In input) {
    Out output;

    float scale = 1.2;
    float rotation_ttime = time * 0.5;
    float3x3 rot = rotation_matrix_y(rotation_ttime) * rotation_matrix_x(rotation_ttime) * rotation_matrix_z(rotation_ttime);

    float3 final_position = position + (rot * scale * input.pos);

    output.position = (matrices) * float4(final_position, 1.0);
    output.normal = input.normal;
    output.uv = input.uv;
    output.position2 = final_position;

    return output;
}
