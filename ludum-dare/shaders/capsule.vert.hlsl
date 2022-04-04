#include "common.h"

[[vk::binding(0)]] cbuffer _ {
    row_major float4x4 matrices;
    float3 player_position;
    float player_facing;
};

struct In {
    float3 pos;
    float3 normal;
    float2 uv;
    uint4 joints;
    float4 weights;
};

struct Out {
    float4 vertex_position: SV_Position;
    float3 normal;
    float2 uv;
    float3 position;
};

[[vk::binding(4)]] cbuffer _ {
    row_major float4x4 joint_transforms[10];
};

Out main(In input) {
    Out output;

    float4x4 skin =
		input.weights.x * joint_transforms[input.joints.x] +
		input.weights.y * joint_transforms[input.joints.y] +
		input.weights.z * joint_transforms[input.joints.z] +
		input.weights.w * joint_transforms[input.joints.w];

    float3x3 rot = rotation_matrix_y(-player_facing);

    float3 final_position = player_position + (rot * (skin * float4(input.pos, 1.0)).xyz);

    output.vertex_position = matrices * float4(final_position, 1.0);
    output.uv = input.uv;
    output.normal = rot * input.normal;
    output.position = final_position;

    return output;
}
