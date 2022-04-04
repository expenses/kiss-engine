#include "common.h"

[[vk::binding(0)]] cbuffer _ {
    float4x4 matrices;
    float3 player_position;
    float player_facing;
};

struct In {
    float3 pos: TEXCOORD0;
    float3 normal: TEXCOORD1;
    float2 uv: TEXCOORD2;
    float4 instance_info: TEXCOORD3;
};

struct Out {
    float4 vertex_position: SV_Position;
    float3 normal: TEXCOORD0;
    float2 uv: TEXCOORD1;
    float3 position: TEXCOORD2;
};

Out main(In input) {
    Out output;

    float scale = 0.75;
    float3 instance_pos = input.instance_info.xyz;
    float instance_rotation = input.instance_info.w;

    float3 final_position = instance_pos + mul(rotation_matrix_y(instance_rotation), scale * input.pos);

    output.vertex_position = mul(matrices, float4(final_position, 1.0));
    output.normal = input.normal;
    output.uv = input.uv;
    output.position = final_position;

    return output;
}
