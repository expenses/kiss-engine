struct In {
    float3 pos;
    float3 normal;
    float3 uv;
};

struct Out {
    float4 position: SV_Position;
    float height;
};

Out main(In input) {
    Out output;

    output.position = float4(float2(input.uv.x, 1.0 - input.uv.y) * 2.0 - 1.0, 0.0, 1.0);

    output.height = input.pos.y;

    return output;
}
