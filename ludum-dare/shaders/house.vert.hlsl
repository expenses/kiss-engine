[[vk::binding(0)]] cbuffer _ {
    row_major float4x4 matrices;
    float3 player_position;
    float player_facing;
};

struct In {
    [[vk::location(0)]] float3 pos: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
    [[vk::location(2)]] float2 uv: TEXCOORD1;
    [[vk::location(3)]] float4 instance_info: TEXCOORD0;
};

struct Out {
    float4 position: SV_Position;
    [[vk::location(0)]] float3 normal: TEXCOORD0;
    [[vk::location(1)]] float2 uv: TEXCOORD1;
    [[vk::location(2)]] float3 position2: TEXCOORD2;
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

    float3 instance_pos = input.instance_info.xyz;
    float instance_rotation = input.instance_info.w;

    float3 final_position = instance_pos + (rotation_matrix_y(instance_rotation) * input.pos);

    output.position = matrices * float4(final_position, 1.0);
    output.normal = input.normal;
    output.uv = input.uv;
    output.position2 = final_position;

    return output;
}
