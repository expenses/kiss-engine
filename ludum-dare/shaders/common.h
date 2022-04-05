#define SUN_DIR normalize(float3(0.0, 10.0, 1.0))

float shadow_factor(float3 position, float3 meteor_position) {
    float2 pos_2d = float2(position.x, position.z);
    float2 meteor_pos_2d = float2(meteor_position.x, meteor_position.z);

    float shadow_scale = (150.0 + position.y - meteor_position.y) * 0.01;

    float ambient = 0.025;

    return max(smoothstep(shadow_scale * 0.9, shadow_scale * 1.1, distance(pos_2d, meteor_pos_2d)), ambient);
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

    static Quaternion from_rotation_x(float angle) {
        return Quaternion::from(sin(angle * 0.5), 0, 0, cos(angle * 0.5));
    }

    static Quaternion from_rotation_y(float angle) {
        return Quaternion::from(0, sin(angle * 0.5), 0, cos(angle * 0.5));
    }

    static Quaternion from_rotation_z(float angle) {
        return Quaternion::from(0, 0, sin(angle * 0.5), cos(angle * 0.5));
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
        return Quaternion::from(
            this.w * other.x + this.x * other.w + this.y * other.z - this.z * other.y,
            this.w * other.y - this.x * other.z + this.y * other.w + this.z * other.x,
            this.w * other.z + this.x * other.y - this.y * other.x + this.z * other.w,
            this.w * other.w - this.x * other.x - this.y * other.y - this.z * other.z
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