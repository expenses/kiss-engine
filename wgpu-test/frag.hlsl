struct In {
    [[vk::location(0)]] float2 uv: TEXCOORD0;
};

struct Out {
    float4 color: SV_TARGET0;
};

[[vk::binding(0)]] cbuffer _ {
    float time;
};

Out main(In input) {
    Out output;

    output.color = float4(input.uv.x, input.uv.y, time % 1.0, 1.0);

    return output;
}
