[[vk::binding(0)]] cbuffer _ {
    row_major float4x4 matrices;
    float3 player_position;
    float player_facing;
    float3 camera_position;
};

struct Ripple {
    float3 position;
    float time;
};

[[vk::binding(1)]] cbuffer _ {
    Ripple ripples[2];
}

struct In {
    [[vk::location(0)]] float3 pos: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
    [[vk::location(2)]] float3 uv: TEXCOORD2;
};

struct Out {
    float4 position: SV_Position;
    [[vk::location(0)]] float2 uv: TEXCOORD0;
    [[vk::location(1)]] float3 normal: TEXCOORD1;
    [[vk::location(2)]] float3 position2: TEXCOORD2;
};


Out main(In input, uint instance_id: SV_InstanceID) {
    Out output;

    Ripple ripple = ripples[instance_id];

    float scale = 0.015 * sqrt(ripple.time);
    float3 position = ripple.position + (scale * input.pos);

    output.position = matrices * float4(position, 1.0);
    output.position2 = position;
    output.uv = input.uv;
    output.normal = input.normal;

    return output;
}