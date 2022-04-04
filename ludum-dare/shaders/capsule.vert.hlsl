#include "common.h"

[[vk::binding(0)]] cbuffer _ {
    Uniforms uniforms;
};

struct In {
    float3 pos: TEXCOORD0;
    float3 normal: TEXCOORD1;
    float2 uv: TEXCOORD2;
    uint4 joints: TEXCOORD3;
    float4 weights: TEXCOORD4;
};

struct Out {
    float4 vertex_position: SV_Position;
    float3 normal: TEXCOORD0;
    float2 uv: TEXCOORD1;
    float3 position: TEXCOORD2;
};

[[vk::binding(4)]] cbuffer _ {
    float4x4 joint_transforms[10];
};

Out main(In input) {
    Out output;

    float4x4 skin =
		input.weights.x * joint_transforms[input.joints.x] +
		input.weights.y * joint_transforms[input.joints.y] +
		input.weights.z * joint_transforms[input.joints.z] +
		input.weights.w * joint_transforms[input.joints.w];

    float3x3 rot = rotation_matrix_y(uniforms.player_facing);

    float3 final_position = uniforms.player_position + mul(rot, mul(skin, float4(input.pos, 1.0)).xyz);

    output.vertex_position = mul(uniforms.matrices, float4(final_position, 1.0));
    output.uv = input.uv;
    output.normal = mul(rot, input.normal);
    output.position = final_position;

    return output;
}
