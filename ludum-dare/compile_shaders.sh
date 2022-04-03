glslc -fshader-stage=vert shaders/plane.vert.hlsl -o             shaders/compiled/plane.vert.spv
glslc -fshader-stage=frag shaders/plane.frag.hlsl -o             shaders/compiled/plane.frag.spv
glslc -fshader-stage=vert shaders/capsule.vert.hlsl -o           shaders/compiled/capsule.vert.spv
glslc -fshader-stage=frag shaders/capsule.frag.hlsl -o           shaders/compiled/capsule.frag.spv
glslc -fshader-stage=vert shaders/bake_height_map.vert.hlsl -o   shaders/compiled/bake_height_map.vert.spv
glslc -fshader-stage=frag shaders/bake_height_map.frag.hlsl -o   shaders/compiled/bake_height_map.frag.spv
glslc -fshader-stage=vert shaders/bounce_sphere.vert.hlsl -o   shaders/compiled/bounce_sphere.vert.spv
glslc -fshader-stage=frag shaders/bounce_sphere.frag.hlsl -o   shaders/compiled/bounce_sphere.frag.spv
glslc -fshader-stage=frag shaders/water.frag.hlsl -o   shaders/compiled/water.frag.spv
glslc -fshader-stage=vert shaders/tree.vert.hlsl -o   shaders/compiled/tree.vert.spv
glslc -fshader-stage=frag shaders/tree.frag.hlsl -o   shaders/compiled/tree.frag.spv
glslc -fshader-stage=vert shaders/house.vert.hlsl -o   shaders/compiled/house.vert.spv

glslc -fshader-stage=vert shaders/meteor.vert.hlsl -o   shaders/compiled/meteor.vert.spv
glslc -fshader-stage=frag shaders/meteor.frag.hlsl -o   shaders/compiled/meteor.frag.spv


glslc -fshader-stage=vert shaders/meteor_outline.vert.hlsl -o   shaders/compiled/meteor_outline.vert.spv
glslc -fshader-stage=frag shaders/meteor_outline.frag.hlsl -o   shaders/compiled/meteor_outline.frag.spv
