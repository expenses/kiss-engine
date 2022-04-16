sh dxc.sh -fvk-invert-y -T  vs_6_7 shaders/plane.vert.hlsl -Fo    shaders/compiled/plane.vert.spv
sh dxc.sh -T ps_6_7 shaders/plane.frag.hlsl -Fo    shaders/compiled/plane.frag.spv
sh dxc.sh -fvk-invert-y -T  vs_6_7 shaders/capsule.vert.hlsl -Fo   shaders/compiled/capsule.vert.spv
sh dxc.sh -T  vs_6_7 shaders/bake_height_map.vert.hlsl -Fo shaders/compiled/bake_height_map.vert.spv
sh dxc.sh -T ps_6_7 shaders/bake_height_map.frag.hlsl -Fo shaders/compiled/bake_height_map.frag.spv
sh dxc.sh -fvk-invert-y -T  vs_6_7 shaders/bounce_sphere.vert.hlsl -Fo shaders/compiled/bounce_sphere.vert.spv
sh dxc.sh -T ps_6_7 shaders/bounce_sphere.frag.hlsl -Fo shaders/compiled/bounce_sphere.frag.spv
sh dxc.sh -T ps_6_7 shaders/water.frag.hlsl -Fo shaders/compiled/water.frag.spv
sh dxc.sh -fvk-invert-y -T  vs_6_7 shaders/tree.vert.hlsl -Fo shaders/compiled/tree.vert.spv
sh dxc.sh -T ps_6_7 shaders/tree.frag.hlsl -Fo shaders/compiled/tree.frag.spv
sh dxc.sh -fvk-invert-y -T  vs_6_7 shaders/house.vert.hlsl -Fo shaders/compiled/house.vert.spv

sh dxc.sh -fvk-invert-y -T  vs_6_7 shaders/meteor.vert.hlsl -Fo shaders/compiled/meteor.vert.spv
sh dxc.sh -T ps_6_7 shaders/meteor.frag.hlsl -Fo shaders/compiled/meteor.frag.spv


sh dxc.sh -fvk-invert-y -T  vs_6_7 shaders/meteor_outline.vert.hlsl -Fo shaders/compiled/meteor_outline.vert.spv
sh dxc.sh -T ps_6_7 shaders/meteor_outline.frag.hlsl -Fo shaders/compiled/meteor_outline.frag.spv
