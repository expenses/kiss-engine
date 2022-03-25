#version 450

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;

layout(location = 0) out vec4 colour;

vec3 tri( vec3 x )
{
    return abs(x-floor(x)-.5);
}
float surfFunc( vec3 p )
{
    float n = dot(tri(p*.15 + tri(p.yzx*.075)), vec3(.444));
    p = p*1.5773 - n;
    p.yz = vec2(p.y + p.z, p.z - p.y) * .866;
    p.xz = vec2(p.x + p.z, p.z - p.x) * .866;
    n += dot(tri(p*.225 + tri(p.yzx*.1125)), vec3(.222));
    return abs(n-.5)*1.9 + (1.-abs(sin(n*9.)))*.05;
}

const vec3 light = normalize(vec3(-1.0, 0.4, 0.9));

void main()
{
    float surfy = surfFunc( v_world_pos );
    float brightness = smoothstep( .2, .3, surfy );

    vec3 normal = normalize(v_normal);

    float lighting = max(dot(normal, light), 0.0);
    float adjusted_lighting = lighting * 0.85 + 0.15;

    colour = vec4( vec3(0.5 + 0.25 * brightness) * (.5+.5*normal) * adjusted_lighting, 1 );
}