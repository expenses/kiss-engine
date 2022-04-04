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
    float3 pos;
    float3 normal;
    float3 uv;
};

struct Out {
    float4 vertex_position: SV_Position;
    float3 normal;
    float2 uv;
    float3 position;
};

Out main(In input) {
    Out output;

    float scale = 1.2;
    float rotation_ttime = time * 0.5;
    float3x3 rot = rotation_matrix_y(rotation_ttime) * rotation_matrix_x(rotation_ttime) * rotation_matrix_z(rotation_ttime);

    float3 final_position = position + (rot * scale * input.pos);

    output.vertex_position = matrices * float4(final_position, 1.0);
    output.normal = input.normal;
    output.uv = input.uv;
    output.position = final_position;

    return output;
}
