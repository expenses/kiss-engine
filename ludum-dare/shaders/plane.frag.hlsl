struct In {
    [[vk::location(0)]] float2 uv: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
};

struct Out {
    float4 color: SV_TARGET0;
};

Out main(In input) {
    Out output;

    output.color = float4(fmod(input.uv, float2(1.0)), 0.0, 1.0);

    return output;
}