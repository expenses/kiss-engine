#include "common.h"

[[vk::binding(0)]] cbuffer _ {
    Uniforms uniforms;
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

    float rotation_time = uniforms.time * 0.5;
    Quaternion rot = Quaternion::from_rotation_x(rotation_time) * Quaternion::from_rotation_y(rotation_time) * Quaternion::from_rotation_z(rotation_time);

    float3 final_position = position + rot * input.pos;

    output.vertex_position = mul(uniforms.matrices, float4(final_position, 1.0));
    output.normal = input.normal;
    output.uv = input.uv;
    output.position = final_position;

    return output;
}
