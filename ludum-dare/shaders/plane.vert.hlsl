[[vk::binding(0)]] cbuffer _ {
    row_major float4x4 matrices;
    float2 player_position;
};

struct In {
    float3 pos;
    float3 normal;
    float3 uv;
};

struct Out {
    float4 vertex_position: SV_Position;
    float2 uv;
    float3 normal;
    float3 position;

};

Out main(In input) {
    Out output;

    output.vertex_position = matrices * float4(input.pos, 1.0);
    output.position = input.pos;
    output.uv = input.uv;
    output.normal = input.normal;

    return output;
}