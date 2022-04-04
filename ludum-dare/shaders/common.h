#define SUN_DIR normalize(float3(0.0, 10.0, 1.0))

float shadow_factor(float3 position, float3 meteor_position) {
    float2 pos_2d = float2(position.x, position.z);
    float2 meteor_pos_2d = float2(meteor_position.x, meteor_position.z);

    float shadow_scale = (150.0 + position.y - meteor_position.y) * 0.01;

    float ambient = 0.025;

    return max(smoothstep(shadow_scale * 0.9, shadow_scale * 1.1, distance(pos_2d, meteor_pos_2d)), ambient);
}

float3x3 rotation_matrix_y(float theta) {
    float cosine = cos(theta);
    float sine = sin(theta);

    return float3x3(
        cosine, 0, sine,
        0, 1, 0,
        -sine, 0, cosine
    );
}

float3x3 rotation_matrix_x(float theta) {
    float cosine = cos(theta);
    float sine = sin(theta);

    return float3x3(
        1, 0, 0,
        0, cosine, -sine,
        0, sine, cosine
    );
}

float3x3 rotation_matrix_z(float theta) {
    float cosine = cos(theta);
    float sine = sin(theta);

    return float3x3(
        cosine, -sine, 0,
        sine, cosine, 0,
        0, 0, 1
    );
}

struct Uniforms {
    float4x4 matrices;
    float3 player_position;
    float player_facing;
    float3 camera_position;
    float time;
    float2 window_size;
};
