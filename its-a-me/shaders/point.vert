#version 450

layout(binding = 0) uniform Uniforms {
    mat4 projection_view;
};

layout(location = 0) in vec3 position;

layout(location = 0) out vec3 colour;

void main()
{
    vec3 pos = position;
    colour = vec3(0.0, 0.0, 0.0);

    if (gl_VertexIndex == 1) {
        pos.y += 1.0;
        colour.g = 0.75;
    }

    gl_Position = projection_view * vec4(pos, 1.0);
}
