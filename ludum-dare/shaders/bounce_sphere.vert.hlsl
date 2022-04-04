#include "common.h"

[[vk::binding(0)]] cbuffer _ {
    Uniforms uniforms;
};

[[vk::binding(1)]] cbuffer _ {
    float3 position;
    float scale;
};

struct In {
    float3 pos: TEXCOORD0;
    float3 normal: TEXCOORD1;
    float3 uv: TEXCOORD2;
};

struct Out {
    float4 position: SV_Position;
    float3 normal: TEXCOORD0;
    float3 dir_to_camera: TEXCOORD1;
};

Out main(In input) {
    Out output;

    float3 final_position = position + (scale * input.pos);

    output.position = mul(uniforms.matrices, float4(final_position, 1.0));
    output.normal = input.normal;
    output.dir_to_camera = uniforms.camera_position - final_position;

    return output;
}
