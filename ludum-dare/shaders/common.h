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

float4x4 scale_matrix(float scale) {
    return float4x4(
        float4(scale, 0.0, 0.0, 0.0),
        float4(0.0, scale, 0.0, 0.0),
        float4(0.0, 0.0, scale, 0.0),
        float4(0.0, 0.0, 0.0, 1.0)
    );
}

float4x4 translation_matrix(float3 translation) {
    return float4x4(
        float4(1.0, 0.0, 0.0, 0.0),
        float4(0.0, 1.0, 0.0, 0.0),
        float4(0.0, 0.0, 1.0, 0.0),
        float4(translation, 1.0)
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

struct Quaternion {
    float x;
    float y;
    float z;
    float w;

    static Quaternion from(float x, float y, float z, float w) {
        Quaternion output;

        output.x = x;
        output.y = y;
        output.z = z;
        output.w = w;

        return output;
    }

    static Quaternion from_rotation_y(float angle) {
        return Quaternion::from(0, sin(angle * 0.5), 0, cos(angle * 0.5));
    }

    Quaternion operator *(float scalar) {
        return Quaternion::from(
            this.x * scalar, this.y * scalar, this.z * scalar, this.w * scalar
        );
    }

    Quaternion operator +(Quaternion other) {
        return Quaternion::from(
            this.x + other.x,
            this.y + other.y,
            this.z + other.z,
            this.w + other.w
        );
    }

    // Adapted from:
    // https://github.com/expenses/dune_scene/blob/127efee150df7a5be7238271f8a05274832c4392/shaders/includes/rotor.glsl#L74-L85
    // in turn adapted from:
    // https://github.com/termhn/ultraviolet/blob/9653d78b68aa19659b904d33d33239bbd2907504/src/rotor.rs#L550-L563
    float3 operator *(float3 vec) {
        float fx = this.w * vec.x - this.z * vec.y + this.y * vec.z;
        float fy = this.w * vec.y + this.z * vec.x - this.x * vec.z;
        float fz = this.w * vec.z - this.y * vec.x + this.x * vec.y;
        float fw = -this.z * vec.z - this.y * vec.y - this.x * vec.x;

        return float3(
            this.w * fx - this.z * fy + this.y * fz - this.x * fw,
            this.w * fy + this.z * fx - this.y * fw - this.x * fz,
            this.w * fz - this.z * fw - this.y * fx + this.x * fy
        );
    }

    // Adapted from:
    // https://github.com/bitshifter/glam-rs/blob/1b703518e7961f9f4e90f40d3969e24462585143/src/core/scalar/quaternion.rs#L69-L81
    Quaternion operator *(Quaternion other) {
        float x0 = this.x;
        float x1 = other.x;
        float y0 = this.y;
        float y1 = other.y;
        float z0 = this.z;
        float z1 = other.z;
        float w0 = this.w;
        float w1 = other.w;

        return Quaternion::from(
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        );
    }
};

struct Similarity {
    float3 translation;
    float scale;
    Quaternion rotation;

    static Similarity from(float3 translation, Quaternion rotation, float scale) {
        Similarity output;

        output.translation = translation;
        output.rotation = rotation;
        output.scale = scale;

        return output;
    }

    Similarity operator *(float scalar) {
        return Similarity::from(
            this.translation * scalar,
            this.rotation * scalar,
            this.scale * scalar
        );
    }

    float3 operator *(float3 vec) {
        return this.translation + (this.scale * (this.rotation * vec));
    }

    Similarity operator *(Similarity child) {
        return Similarity::from(
            this * child.translation,
            this.rotation * child.rotation,
            this.scale * child.scale
        );
    }

    Similarity operator +(Similarity other) {
        return Similarity::from(
            this.translation + other.translation,
            this.rotation + other.rotation,
            this.scale + other.scale
        );
    }
};