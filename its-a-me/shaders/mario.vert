#version 450

layout(binding = 0) uniform Uniforms {
    mat4 projection_view;
};

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 colour;
layout(location = 3) in vec3 normal;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec3 v_colour;
layout(location = 2) out vec3 v_normal;

void main()
{
    v_uv = uv;
    v_colour = colour;
    v_normal = normal;

    gl_Position = projection_view * vec4(position / 50.0, 1.0);
}
