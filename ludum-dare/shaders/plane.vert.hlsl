[[vk::binding(0)]] cbuffer _ {
    float4x4 matrices;
    float2 player_position;
};

struct In {
    float3 pos: TEXCOORD0;
    float3 normal: TEXCOORD1;
    float2 uv: TEXCOORD2;
};

struct Out {
    float4 vertex_position: SV_Position;
    float2 uv: TEXCOORD0;
    float3 normal: TEXCOORD1;
    float3 position: TEXCOORD2;
};

Out main(In input) {
    Out output;

    output.vertex_position = mul(matrices, float4(input.pos, 1.0));
    output.position = input.pos;
    output.uv = input.uv;
    output.normal = input.normal;

    return output;
}