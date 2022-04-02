[[vk::binding(0)]] cbuffer _ {
    row_major float4x4 matrices;
    float3 player_position;
    float player_facing;
};

[[vk::binding(1)]] cbuffer _ {
    float3 position;
    float scale;
};

struct In {
    [[vk::location(0)]] float3 pos: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
};

struct Out {
    float4 position: SV_Position;
    [[vk::location(0)]] float3 normal: TEXCOORD1;
};

Out main(In input) {
    Out output;

    float3 final_position = position + (scale * input.pos);


    output.position = (matrices) * float4(final_position, 1.0);
    output.normal = input.normal;

    return output;
}
