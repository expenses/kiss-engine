struct Out {
    float4 position: SV_Position;
    [[vk::location(0)]] float2 uv: TEXCOORD0;
};

Out main(uint vertex_id: SV_VertexID) {
    Out output;

    output.uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    output.position = float4(output.uv * 2.0 + -1.0, 1.0, 1.0);

    return output;
}