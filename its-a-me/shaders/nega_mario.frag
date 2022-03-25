#version 450

layout(binding = 1) uniform sampler samp;

layout(binding = 2) uniform texture2D tex;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec3 v_colour;
layout(location = 2) in vec3 v_normal;

layout(location = 0) out vec4 colour;

const vec3 light = normalize(vec3(-1.0, 0.4, 0.9));

void main() 
{
    vec3 normal = normalize(v_normal);

    float lighting = max(dot(normal, light), 0.0);
    float adjusted_lighting = lighting * 0.85 + 0.15;

    vec4 tex_colour = texture(sampler2D(tex, samp), v_uv);

    vec3 mixed_colour = mix(v_colour, tex_colour.rgb, tex_colour.a);

    colour = vec4((1.0 - mixed_colour) * adjusted_lighting, 1.0);
}