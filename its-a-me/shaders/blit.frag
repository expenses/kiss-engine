#version 450

layout(binding = 0) uniform sampler samp;

layout(binding = 1) uniform texture2D tex;

layout (location = 0) in vec2 uv;

layout(location = 0) out vec4 colour;

void main()
{
    colour = texture(sampler2D(tex, samp), uv);
}