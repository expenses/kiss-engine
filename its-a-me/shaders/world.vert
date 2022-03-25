#version 450

layout(binding = 0) uniform Uniforms {
    mat4 projection_view;
};

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal;

void main()
{
    v_world_pos = position;
    v_normal = normal;
    gl_Position = projection_view * vec4(position, 1.0);
}
