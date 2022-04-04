#include "common.h"

[[vk::binding(0)]] cbuffer _ {
    float4x4 matrices;
    float3 player_position;
    float player_facing;
    float3 camera_position;
    float time;
};

[[vk::binding(1)]] cbuffer _ {
    float3 position;
};

struct In {
    float3 pos: TEXCOORD0;
    float3 normal: TEXCOORD1;
    float2 uv: TEXCOORD2;
};

struct Out {
    float4 vertex_position: SV_Position;
    float3 normal: TEXCOORD0;
    float2 uv: TEXCOORD1;
    float3 position: TEXCOORD2;

};

Out main(In input) {
    Out output;

    float rotation_time = time * 0.5;
    float3x3 rot = rotation_matrix_y(rotation_time) * rotation_matrix_x(rotation_time) * rotation_matrix_z(rotation_time);

    float3 final_position = position + mul(rot, input.pos);

    output.vertex_position = mul(matrices, float4(final_position, 1.0));
    output.normal = input.normal;
    output.uv = input.uv;
    output.position = final_position;

    return output;
}
