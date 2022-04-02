[[vk::binding(0)]] cbuffer _ {
    row_major float4x4 matrices;
    float3 player_position;
};

struct In {
    [[vk::location(0)]] float3 pos: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
};

struct Out {
    float4 position: SV_Position;
    [[vk::location(0)]] float2 uv: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
};

Out main(In input) {
    Out output;

    output.position = (matrices * float4(player_position + input.pos, 1.0));
    output.uv = float2(input.pos.x, input.pos.z);
    output.normal = input.normal;

    return output;
}