struct In {
    float3 pos: TEXCOORD0;
    float3 normal: TEXCOORD1;
    float2 uv: TEXCOORD2;
};

struct Out {
    float4 position: SV_Position;
    float height: TEXCOORD0;
};

Out main(In input) {
    Out output;

    output.position = float4(float2(input.uv.x, input.uv.y) * 2.0 - 1.0, 0.0, 1.0);

    output.height = input.pos.y;

    return output;
}
