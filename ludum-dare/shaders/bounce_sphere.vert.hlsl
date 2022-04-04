[[vk::binding(0)]] cbuffer _ {
    row_major float4x4 matrices;
    float3 player_position;
    float player_facing;
    float3 camera_position;
};

[[vk::binding(1)]] cbuffer _ {
    float3 position;
    float scale;
};

struct In {
    float3 pos;
    float3 normal;
    float3 uv;
};

struct Out {
    float4 position: SV_Position;
    float3 normal;
    float3 dir_to_camera;
};

Out main(In input) {
    Out output;

    float3 final_position = position + (scale * input.pos);

    output.position = matrices * float4(final_position, 1.0);
    output.normal = input.normal;
    output.dir_to_camera = camera_position - final_position;

    return output;
}
