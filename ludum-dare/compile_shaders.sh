glslc -g -fshader-stage=vert shaders/plane.vert.hlsl -o             shaders/compiled/plane.vert.spv
glslc -g -fshader-stage=vert shaders/capsule.vert.hlsl -o           shaders/compiled/capsule.vert.spv
glslc -g -fshader-stage=frag shaders/plane.frag.hlsl -o             shaders/compiled/plane.frag.spv
glslc -g -fshader-stage=vert shaders/bake_height_map.vert.hlsl -o   shaders/compiled/bake_height_map.vert.spv
glslc -g -fshader-stage=frag shaders/bake_height_map.frag.hlsl -o   shaders/compiled/bake_height_map.frag.spv

glslc -g -fshader-stage=vert shaders/bounce_sphere.vert.hlsl -o   shaders/compiled/bounce_sphere.vert.spv
glslc -g -fshader-stage=frag shaders/bounce_sphere.frag.hlsl -o   shaders/compiled/bounce_sphere.frag.spv
