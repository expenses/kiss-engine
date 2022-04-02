[[vk::binding(0)]] cbuffer _ {
    row_major float4x4 matrices;
    float3 player_position;
    float player_facing;
};

[[vk::binding(1)]] Texture2D<float> depth_map_tex;
[[vk::binding(2)]] sampler tex_sampler;

struct In {
    [[vk::location(0)]] float3 pos: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
};

struct Out {
    float4 position: SV_Position;
    [[vk::location(0)]] float2 uv: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
};


row_major float3x3 rotation_matrix_y(float theta) {
    float cos = cos(theta);
    float sin = sin(theta);

    return float3x3(
        cos, 0, sin,
        0, 1, 0,
        -sin, 0, cos
    );
}

Out main(In input) {
    Out output;

    float3 p = player_position;

    float3x3 rot = rotation_matrix_y(-player_facing);


    output.position = (matrices) * float4(p + (rot * input.pos), 1.0);
    output.uv = float2(input.pos.x, input.pos.z);
    output.normal = rot * input.normal;

    return output;
}
