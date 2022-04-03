struct In {
    [[vk::location(0)]] float3 pos: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
    [[vk::location(2)]] float3 uv: TEXCOORD2;
};

struct Out {
    float4 position: SV_Position;
    [[vk::location(0)]] float height: TEXCOORD0;
};

Out main(In input) {
    Out output;

    output.position = float4(float2(input.uv.x, 1.0 - input.uv.y) * 2.0 - 1.0, 0.0, 1.0);

    output.height = input.pos.y;

    return output;
}
