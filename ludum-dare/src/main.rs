use glam::Mat3;
use glam::Mat4;
use glam::Quat;
use glam::Vec2;
use glam::Vec3;
use glam::Vec4;
use kiss_engine_wgpu::{
    BindGroupLayoutSettings, BindingResource, Device, RenderPipelineDesc, VertexBufferLayout,
};
use rand::Rng;
use std::ops::*;

mod animation;
mod assets;

use assets::{load_image, Model};

use wgpu::util::DeviceExt;
use winit::{
    event::{self, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

pub const Z_NEAR: f32 = 0.01;
pub const Z_FAR: f32 = 50_000.0;

const BASIC_FORMAT: &[VertexBufferLayout] = &[
    VertexBufferLayout {
        location: 0,
        format: wgpu::VertexFormat::Float32x3,
        step_mode: wgpu::VertexStepMode::Vertex,
    },
    VertexBufferLayout {
        location: 1,
        format: wgpu::VertexFormat::Float32x3,
        step_mode: wgpu::VertexStepMode::Vertex,
    },
    VertexBufferLayout {
        location: 2,
        format: wgpu::VertexFormat::Float32x2,
        step_mode: wgpu::VertexStepMode::Vertex,
    },
];

const INSTANCED_FORMAT: &[VertexBufferLayout] = &[
    VertexBufferLayout {
        location: 0,
        format: wgpu::VertexFormat::Float32x3,
        step_mode: wgpu::VertexStepMode::Vertex,
    },
    VertexBufferLayout {
        location: 1,
        format: wgpu::VertexFormat::Float32x3,
        step_mode: wgpu::VertexStepMode::Vertex,
    },
    VertexBufferLayout {
        location: 2,
        format: wgpu::VertexFormat::Float32x2,
        step_mode: wgpu::VertexStepMode::Vertex,
    },
    VertexBufferLayout {
        location: 3,
        format: wgpu::VertexFormat::Float32x4,
        step_mode: wgpu::VertexStepMode::Instance,
    },
];

fn perspective_matrix_reversed(width: u32, height: u32) -> glam::Mat4 {
    let aspect_ratio = width as f32 / height as f32;
    let vertical_fov = 59.0_f32.to_radians();

    let focal_length_y = 1.0 / (vertical_fov / 2.0).tan();
    let focal_length_x = focal_length_y / aspect_ratio;

    let near_minus_far = Z_NEAR - Z_FAR;

    glam::Mat4::from_cols(
        glam::Vec4::new(focal_length_x, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, focal_length_y, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, -Z_FAR / near_minus_far - 1.0, -1.0),
        glam::Vec4::new(0.0, 0.0, -Z_NEAR * Z_FAR / near_minus_far, 0.0),
    )
}

async fn run() {
    #[cfg(feature = "wasm")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));

        let level: log::Level = wasm_web_helpers::parse_url_query_string_from_window("RUST_LOG")
            .map(|x| x.parse().ok())
            .flatten()
            .unwrap_or(log::Level::Info);
        console_log::init_with_level(level).expect("could not initialize logger");
    }

    #[cfg(not(feature = "wasm"))]
    env_logger::init();

    let event_loop = EventLoop::new();

    let window = winit::window::WindowBuilder::new()
        .build(&event_loop)
        .expect("Failed to create window");

    #[cfg(feature = "wasm")]
    {
        // On wasm, append the canvas to the document body
        wasm_web_helpers::append_canvas(&window);
    }

    let backend = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);

    let instance = wgpu::Instance::new(backend);
    let size = window.inner_size();
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .expect("No suitable GPU adapters found on the system!");

    let adapter_info = adapter.get_info();
    println!(
        "Using {} with the {:?} backend",
        adapter_info.name, adapter_info.backend
    );

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("device"),
                features: Default::default(),
                #[cfg(feature = "wasm")]
                limits: wgpu::Limits::downlevel_webgl2_defaults(),
                #[cfg(not(feature = "wasm"))]
                limits: Default::default(),
            },
            None,
        )
        .await
        .expect("Unable to find a suitable GPU adapter!");

    let (plane, _) = Model::new(include_bytes!("../plane.glb"), &device, "plane");
    let (capsule, mut player_joints) =
        Model::new(include_bytes!("../fire_giant.glb"), &device, "capsule");
    let (bounce_sphere, _) = Model::new(
        include_bytes!("../bounce_sphere.glb"),
        &device,
        "bounce_sphere",
    );
    let (water, _) = Model::new(include_bytes!("../water.glb"), &device, "water");
    let (tree, _) = Model::new(include_bytes!("../tree.glb"), &device, "tree");
    let (house, _) = Model::new(include_bytes!("../house.glb"), &device, "house");

    let (meteor, _) = Model::new(include_bytes!("../meteor.glb"), &device, "meteor");

    let mut perspective_matrix = perspective_matrix_reversed(size.width, size.height);

    let size = window.inner_size();

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface
            .get_preferred_format(&adapter)
            .expect("Failed to get preferred surface format"),
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
    };
    surface.configure(&device, &config);

    // Prepare glyph_brush
    let font = wgpu_glyph::ab_glyph::FontArc::try_from_slice(include_bytes!(
        "../VonwaonBitmap-16pxLite.ttf"
    ))
    .expect("Failed to load font");

    let mut glyph_brush = wgpu_glyph::GlyphBrushBuilder::using_font(font)
        .initial_cache_size((512, 512))
        .build(&device, config.format);

    let mut staging_belt = wgpu::util::StagingBelt::new(1024);

    let mut device = Device::new(device);

    let mut state = State::default();

    let player_height_offset = Vec3::new(0.0, 1.5, 0.0);

    let view_matrix = {
        Mat4::look_at_rh(
            Vec3::new(state.player_position.x, 0.0, state.player_position.y)
                + player_height_offset
                + state.orbit.as_vector(),
            Vec3::new(state.player_position.x, 1.0, state.player_position.y) + player_height_offset,
            Vec3::Y,
        )
    };

    let uniform_buffer = device.create_resource(device.inner.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("uniform buffer"),
            contents: bytemuck::bytes_of(&Uniforms {
                matrices: perspective_matrix * view_matrix,
                player_position: Vec3::new(state.player_position.x, 0.0, state.player_position.y),
                player_facing: state.player_facing,
                camera_position: Vec3::new(state.player_position.x, 0.0, state.player_position.y)
                    + player_height_offset
                    + state.orbit.as_vector(),
                time: state.time,
                window_size: Vec2::new(config.width as f32, config.height as f32),
                ..Default::default()
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    ));

    let meteor_buffer = device.create_resource(device.inner.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("meteor buffer"),
            contents: bytemuck::bytes_of(&MeteorGpuProps {
                position: state.meteor_props.position,
                _padding: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    ));

    let bounce_sphere_buffer = device.create_resource(device.inner.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("bounce sphere buffer buffer"),
            contents: bytemuck::bytes_of(&state.bounce_sphere_props),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    ));

    let height_map = device.get_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: 1024,
            height: 1024,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // Has to be rgba float because of webgl.
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        label: Some("height map"),
    });

    let buffer_size = 1024 * 1024 * 4 * 4;

    let target_buffer = device.inner.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    {
        let mut encoder = device
            .inner
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("init encoder"),
            });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("init render pass"),
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &height_map.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });

        let device = device.with_formats(&[wgpu::TextureFormat::Rgba32Float], None);

        let pipeline = device.get_pipeline(
            "bake pipeline",
            device.device.get_shader(
                "shaders/compiled/bake_height_map.vert.spv",
                #[cfg(feature = "standalone")]
                include_bytes!("../shaders/compiled/bake_height_map.vert.spv"),
                Default::default(),
            ),
            device.device.get_shader(
                "shaders/compiled/bake_height_map.frag.spv",
                #[cfg(feature = "standalone")]
                include_bytes!("../shaders/compiled/bake_height_map.frag.spv"),
                Default::default(),
            ),
            RenderPipelineDesc {
                primitive: wgpu::PrimitiveState::default(),
                ..Default::default()
            },
            BASIC_FORMAT,
        );

        let bind_group = device.get_bind_group("bake height map", pipeline, &[]);

        render_pass.set_pipeline(&pipeline.pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);

        plane.render(&mut render_pass, 1);

        drop(render_pass);

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &height_map.texture,
                mip_level: 0,
                origin: Default::default(),
                aspect: Default::default(),
            },
            wgpu::ImageCopyBuffer {
                buffer: &target_buffer,
                layout: wgpu::ImageDataLayout {
                    bytes_per_row: Some(
                        std::num::NonZeroU32::new(1024 * 4 * 4).expect("unreachable"),
                    ),
                    ..Default::default()
                },
            },
            wgpu::Extent3d {
                width: 1024,
                height: 1024,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(std::iter::once(encoder.finish()));
    }

    let mut rng = rand::rngs::OsRng::default();

    let heightmap = {
        let slice = target_buffer.slice(..);

        let map_future = slice.map_async(wgpu::MapMode::Read);

        device.inner.poll(wgpu::Maintain::Wait);

        map_future.await.expect("Mapping height map slice failed");

        let bytes = slice.get_mapped_range();

        let vec4s: &[Vec4] = bytemuck::cast_slice(&bytes);

        HeightMap {
            floats: vec4s.iter().map(|vec4| vec4.x).collect(),
            height: 1024,
            width: 1024,
        }
    };

    target_buffer.unmap();

    drop(target_buffer);

    let forest_map = {
        let image = image::load_from_memory(include_bytes!("../forestmap.png"))
            .expect("Failed to read image")
            .to_rgb32f();

        HeightMap {
            floats: image.pixels().map(|pixel| pixel.0[0]).collect(),
            height: image.height(),
            width: image.height(),
        }
    };

    let mut forest_points = Vec::new();

    let infinite_loop_cap = 10000;

    // Bound this loop as it could go infinite otherwise.
    // I'm doing this because I had some issues with the heightmap on webgl.
    for _ in 0..infinite_loop_cap {
        if forest_points.len() < 1000 {
            let x = rng.gen_range(0.0..1.0);
            let y = rng.gen_range(0.0..1.0);

            let value = rng.gen_range(0.0..1.0);

            let heightmap_pos = heightmap.sample(Vec2::new(x, y));

            if forest_map.sample(Vec2::new(x, y)) > value
                && heightmap_pos < 4.5
                && heightmap_pos > 0.25
            {
                forest_points.push(Vec4::new(
                    (x - 0.5) * 80.0,
                    heightmap_pos,
                    (y - 0.5) * 80.0,
                    rng.gen_range(0.6..0.8),
                ));
            }
        }
    }

    let mut house_points = Vec::new();

    for _ in 0..infinite_loop_cap {
        if house_points.len() < 50 {
            let x = rng.gen_range(0.0..1.0);
            let y = rng.gen_range(0.0..1.0);

            let value = rng.gen_range(0.0..1.0);

            let heightmap_pos = heightmap.sample(Vec2::new(x, y));

            if forest_map.sample(Vec2::new(x, y)) < value
                && heightmap_pos < 4.5
                && heightmap_pos > 0.25
            {
                house_points.push(Vec4::new(
                    (x - 0.5) * 80.0,
                    heightmap_pos,
                    (y - 0.5) * 80.0,
                    rng.gen_range(0.0..std::f32::consts::TAU),
                ));
            }
        }
    }

    let num_forest_points = forest_points.len() as u32;
    let num_house_points = house_points.len() as u32;

    let forest_points = device.create_resource(device.inner.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("forest buffer"),
            contents: bytemuck::cast_slice(&forest_points),
            usage: wgpu::BufferUsages::VERTEX,
        },
    ));

    let house_points = device.create_resource(device.inner.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("house buffer"),
            contents: bytemuck::cast_slice(&house_points),
            usage: wgpu::BufferUsages::VERTEX,
        },
    ));

    let grass_texture = load_image(
        &device,
        &queue,
        include_bytes!("../grass.png"),
        "grass texture",
    );
    let sand_texture = load_image(
        &device,
        &queue,
        include_bytes!("../sand.png"),
        "sand texture",
    );
    let rock_texture = load_image(
        &device,
        &queue,
        include_bytes!("../rock.png"),
        "rock texture",
    );
    let forest_texture = load_image(
        &device,
        &queue,
        include_bytes!("../grass_forest.png"),
        "forest texture",
    );
    let house_texture = load_image(
        &device,
        &queue,
        include_bytes!("../house.png"),
        "house texture",
    );

    let forest_map_tex = load_image(
        &device,
        &queue,
        include_bytes!("../forestmap.png"),
        "forest map texture",
    );

    let giant_tex = load_image(
        &device,
        &queue,
        include_bytes!("../fire_giant.png"),
        "fire giant texture",
    );

    let meteor_tex = load_image(
        &device,
        &queue,
        include_bytes!("../meteor.png"),
        "meteor texture",
    );

    let sampler = device.create_resource(device.inner.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        ..Default::default()
    }));

    let linear_sampler =
        device.create_resource(device.inner.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

    let joint_transforms_buffer =
        device.create_resource(device.inner.create_buffer(&wgpu::BufferDescriptor {
            label: Some("player joint transforms"),
            size: (capsule.joint_indices_to_node_indices.len() * std::mem::size_of::<Mat4>())
                as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        }));

    event_loop.run(move |event, _, control_flow| match event {
        event::Event::WindowEvent {
            event:
                WindowEvent::Resized(size)
                | WindowEvent::ScaleFactorChanged {
                    new_inner_size: &mut size,
                    ..
                },
            ..
        } => {
            println!("Resizing to {:?}", size);
            config.width = size.width.max(1);
            config.height = size.height.max(1);
            surface.configure(&device.inner, &config);
            perspective_matrix = perspective_matrix_reversed(size.width, size.height);
        }
        event::Event::WindowEvent { event, .. } => match event {
            WindowEvent::KeyboardInput {
                input:
                    event::KeyboardInput {
                        virtual_keycode: Some(event::VirtualKeyCode::Escape),
                        state: event::ElementState::Pressed,
                        ..
                    },
                ..
            }
            | WindowEvent::CloseRequested => {
                *control_flow = ControlFlow::Exit;
            }
            WindowEvent::KeyboardInput {
                input:
                    event::KeyboardInput {
                        virtual_keycode: Some(key),
                        state: key_state,
                        ..
                    },
                ..
            } => {
                let pressed = key_state == event::ElementState::Pressed;

                match key {
                    VirtualKeyCode::W | VirtualKeyCode::Up => state.kbd.up = pressed,
                    VirtualKeyCode::A | VirtualKeyCode::Left => state.kbd.left = pressed,
                    VirtualKeyCode::S | VirtualKeyCode::Down => state.kbd.down = pressed,
                    VirtualKeyCode::D | VirtualKeyCode::Right => state.kbd.right = pressed,
                    VirtualKeyCode::C if pressed => {
                        state.orbit.yaw = state.player_facing;
                    }
                    VirtualKeyCode::Space if pressed && state.animation_id != 1 => {
                        state.animation_id = 1;
                        state.animation_time = 0.0;
                    }
                    VirtualKeyCode::R if pressed && state.lost => {
                        state = State::default();
                    }
                    _ => {}
                }
            }
            WindowEvent::Focused(false) => {
                state.kbd = Default::default();
            }
            _ => {}
        },
        event::Event::MainEventsCleared => {
            let current_facing = { state.orbit.yaw };

            let forwards = state.kbd.up as i32 - state.kbd.down as i32;
            let right = -(state.kbd.right as i32 - state.kbd.left as i32);

            let delta_time = 1.0 / 60.0;

            state.time += delta_time;

            if (forwards, right) != (0, 0) {
                if state.animation_id == 0 {
                    state.animation_id = 2;
                    state.animation_time = 0.0;
                }

                let new_player_facing = current_facing
                    + match (forwards, right) {
                        (0, 1) => 90.0_f32.to_radians(),
                        (0, -1) => -90.0_f32.to_radians(),
                        (1, 1) => 45.0_f32.to_radians(),
                        (1, -1) => -45.0_f32.to_radians(),

                        (-1, -1) => -135.0_f32.to_radians(),
                        (-1, 1) => 135.0_f32.to_radians(),

                        (-1, 0) => 180.0_f32.to_radians(),

                        _ => 0.0,
                    };

                state.player_facing +=
                    short_angle_dist(state.player_facing, new_player_facing) * 0.5;

                state.orbit.yaw += short_angle_dist(state.orbit.yaw, state.player_facing) * 0.015;

                state.player_speed = (state.player_speed + delta_time).min(1.0);
            } else {
                state.player_speed *= 0.9;

                if state.animation_id == 2 {
                    state.animation_id = 0;
                    state.animation_time = 0.0;
                }
            }

            let movement = Quat::from_rotation_y(state.player_facing)
                * Vec3::new(0.0, 0.0, -delta_time * 10.0 * state.player_speed);

            state.player_position.x += movement.x;
            state.player_position.y += movement.z;

            state.player_position.x = state.player_position.x.max(-40.0).min(40.0);
            state.player_position.y = state.player_position.y.max(-40.0).min(40.0);

            let player_position_3d = Vec3::new(
                state.player_position.x,
                heightmap.sample(state.player_position / 80.0 + 0.5),
                state.player_position.y,
            );

            state.bounce_sphere_props.scale += state.bounce_sphere_props.scale * delta_time * 0.5;

            if state.bounce_sphere_props.scale > 1.5 {
                state.bounce_sphere_props.scale = 0.0;
            }

            state.meteor_props.velocity.y -= delta_time * 5.0;

            if (state.meteor_props.position.x + state.meteor_props.velocity.x * delta_time).abs()
                > 40.0
            {
                state.meteor_props.velocity.x *= -1.0;
            }

            if (state.meteor_props.position.z + state.meteor_props.velocity.z * delta_time).abs()
                > 40.0
            {
                state.meteor_props.velocity.z *= -1.0;
            }

            state.meteor_props.position += state.meteor_props.velocity * delta_time;

            if state.bounce_sphere_props.scale > 0.0
                && state
                    .meteor_props
                    .position
                    .distance(state.bounce_sphere_props.position)
                    < (state.bounce_sphere_props.scale + 1.0)
            {
                // This code can run move than once per bounce so we just do this hack lol
                if state.meteor_props.velocity.y <= 0.0 {
                    state.bounces += 1;
                }

                state.meteor_props.velocity.y = state.meteor_props.velocity.y.abs();

                let rotation = rng.gen_range(0.0..=std::f32::consts::TAU);
                let velocity = rng.gen_range(1.0..=5.0);

                state.meteor_props.velocity.x = velocity * rotation.cos();
                state.meteor_props.velocity.z = velocity * rotation.sin();
            } else if state.meteor_props.position.y < 0.0 {
                state.lost = true;

                state.meteor_props.position.y -= 100.0;
            }

            if state.animation_id == 1
                && state.animation_time > 0.25
                && state.animation_time < 0.75
                && state.bounce_sphere_props.scale == 0.0
            {
                let player_position = Vec3::new(
                    state.player_position.x,
                    heightmap.sample(state.player_position / 80.0 + 0.5),
                    state.player_position.y,
                );

                state.bounce_sphere_props.scale = 1.0;
                state.bounce_sphere_props.position = player_position + Vec3::Y * 4.0;
            }

            state.animation_time += delta_time;

            while state.animation_time > capsule.animations[state.animation_id].total_time() {
                state.animation_time -= capsule.animations[state.animation_id].total_time();

                if state.animation_id == 1 {
                    state.animation_id = 0;
                }
            }

            capsule.animations[state.animation_id].animate(
                &mut player_joints,
                state.animation_time,
                &capsule.depth_first_nodes,
            );

            let joint_transforms: Vec<_> = player_joints
                .iter(
                    &capsule.joint_indices_to_node_indices,
                    &capsule.inverse_bind_matrices,
                )
                .collect();

            queue.write_buffer(
                &joint_transforms_buffer,
                0,
                bytemuck::cast_slice(&joint_transforms),
            );

            queue.write_buffer(
                &meteor_buffer,
                0,
                bytemuck::bytes_of(&MeteorGpuProps {
                    position: state.meteor_props.position,
                    _padding: 0,
                }),
            );

            queue.write_buffer(
                &bounce_sphere_buffer,
                0,
                bytemuck::bytes_of(&state.bounce_sphere_props),
            );

            let view_matrix = {
                Mat4::look_at_rh(
                    player_position_3d + player_height_offset + state.orbit.as_vector(),
                    player_position_3d + player_height_offset,
                    Vec3::Y,
                )
            };

            queue.write_buffer(
                &uniform_buffer,
                0,
                bytemuck::bytes_of(&Uniforms {
                    matrices: perspective_matrix * view_matrix,
                    player_position: player_position_3d,
                    player_facing: state.player_facing,
                    camera_position: player_position_3d
                        + player_height_offset
                        + state.orbit.as_vector(),
                    time: state.time,
                    window_size: Vec2::new(config.width as f32, config.height as f32),
                    ..Default::default()
                }),
            );

            window.request_redraw();
        }
        event::Event::RedrawRequested(_) => {
            let frame = match surface.get_current_texture() {
                Ok(frame) => frame,
                Err(_) => {
                    surface.configure(&device.inner, &config);
                    surface
                        .get_current_texture()
                        .expect("Failed to acquire next surface texture!")
                }
            };
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            let mut encoder =
                device
                    .inner
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("command encoder"),
                    });

            let depth_texture = device.get_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d {
                    width: config.width,
                    height: config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                label: Some("depth texture"),
            });

            let opaque_texture = device.get_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d {
                    width: config.width,
                    height: config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                label: Some("opaque texture"),
            });

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main render pass"),
                color_attachments: &[
                    wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
                                a: 1.0,
                            }),
                            store: true,
                        },
                    },
                    wgpu::RenderPassColorAttachment {
                        view: &opaque_texture.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
                                a: 1.0,
                            }),
                            store: true,
                        },
                    },
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            {
                let formats = &[config.format, wgpu::TextureFormat::Rgba8UnormSrgb];

                let device = device.with_formats(formats, Some(wgpu::TextureFormat::Depth32Float));

                let pipeline = device.get_pipeline(
                    "plane pipeline",
                    device.device.get_shader(
                        "shaders/compiled/plane.vert.spv",
                        #[cfg(feature = "standalone")]
                        include_bytes!("../shaders/compiled/plane.vert.spv"),
                        Default::default(),
                    ),
                    device.device.get_shader(
                        "shaders/compiled/plane.frag.spv",
                        #[cfg(feature = "standalone")]
                        include_bytes!("../shaders/compiled/plane.frag.spv"),
                        Default::default(),
                    ),
                    RenderPipelineDesc {
                        ..Default::default()
                    },
                    BASIC_FORMAT,
                );

                let bind_group = device.get_bind_group(
                    "plane",
                    pipeline,
                    &[
                        BindingResource::Buffer(&uniform_buffer),
                        BindingResource::Buffer(&meteor_buffer),
                        BindingResource::Sampler(&sampler),
                        BindingResource::Sampler(&linear_sampler),
                        BindingResource::Texture(&grass_texture),
                        BindingResource::Texture(&sand_texture),
                        BindingResource::Texture(&rock_texture),
                        BindingResource::Texture(&forest_texture),
                        BindingResource::Texture(&forest_map_tex),
                    ],
                );

                render_pass.set_pipeline(&pipeline.pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);

                plane.render(&mut render_pass, 1);

                let pipeline = device.get_pipeline(
                    "player pipeline",
                    device.device.get_shader(
                        "shaders/compiled/capsule.vert.spv",
                        #[cfg(feature = "standalone")]
                        include_bytes!("../shaders/compiled/capsule.vert.spv"),
                        Default::default(),
                    ),
                    device.device.get_shader(
                        "shaders/compiled/tree.frag.spv",
                        #[cfg(feature = "standalone")]
                        include_bytes!("../shaders/compiled/tree.frag.spv"),
                        Default::default(),
                    ),
                    RenderPipelineDesc {
                        ..Default::default()
                    },
                    &[
                        VertexBufferLayout {
                            location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                            step_mode: wgpu::VertexStepMode::Vertex,
                        },
                        VertexBufferLayout {
                            location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                            step_mode: wgpu::VertexStepMode::Vertex,
                        },
                        VertexBufferLayout {
                            location: 2,
                            format: wgpu::VertexFormat::Float32x2,
                            step_mode: wgpu::VertexStepMode::Vertex,
                        },
                        VertexBufferLayout {
                            location: 3,
                            format: wgpu::VertexFormat::Uint16x4,
                            step_mode: wgpu::VertexStepMode::Vertex,
                        },
                        VertexBufferLayout {
                            location: 4,
                            format: wgpu::VertexFormat::Float32x4,
                            step_mode: wgpu::VertexStepMode::Vertex,
                        },
                    ],
                );

                let bind_group = device.get_bind_group(
                    "capsule",
                    pipeline,
                    &[
                        BindingResource::Buffer(&uniform_buffer),
                        BindingResource::Buffer(&meteor_buffer),
                        BindingResource::Sampler(&sampler),
                        BindingResource::Texture(&giant_tex),
                        BindingResource::Buffer(&joint_transforms_buffer),
                    ],
                );

                render_pass.set_pipeline(&pipeline.pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);

                render_pass.set_vertex_buffer(3, capsule.joints.slice(..));
                render_pass.set_vertex_buffer(4, capsule.weights.slice(..));
                capsule.render(&mut render_pass, 1);

                let pipeline = device.get_pipeline(
                    "meteor pipeline",
                    device.device.get_shader(
                        "shaders/compiled/meteor.vert.spv",
                        #[cfg(feature = "standalone")]
                        include_bytes!("../shaders/compiled/meteor.vert.spv"),
                        Default::default(),
                    ),
                    device.device.get_shader(
                        "shaders/compiled/meteor.frag.spv",
                        #[cfg(feature = "standalone")]
                        include_bytes!("../shaders/compiled/meteor.frag.spv"),
                        Default::default(),
                    ),
                    RenderPipelineDesc {
                        ..Default::default()
                    },
                    BASIC_FORMAT,
                );

                let bind_group = device.get_bind_group(
                    "meteor",
                    pipeline,
                    &[
                        BindingResource::Buffer(&uniform_buffer),
                        BindingResource::Buffer(&meteor_buffer),
                        BindingResource::Sampler(&sampler),
                        BindingResource::Texture(&meteor_tex),
                    ],
                );

                render_pass.set_pipeline(&pipeline.pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);

                meteor.render(&mut render_pass, 1);

                {
                    let pipeline = device.get_pipeline(
                        "trees pipeline",
                        device.device.get_shader(
                            "shaders/compiled/tree.vert.spv",
                            #[cfg(feature = "standalone")]
                            include_bytes!("../shaders/compiled/tree.vert.spv"),
                            Default::default(),
                        ),
                        device.device.get_shader(
                            "shaders/compiled/tree.frag.spv",
                            #[cfg(feature = "standalone")]
                            include_bytes!("../shaders/compiled/tree.frag.spv"),
                            Default::default(),
                        ),
                        RenderPipelineDesc {
                            ..Default::default()
                        },
                        INSTANCED_FORMAT,
                    );

                    let bind_group = device.get_bind_group(
                        "trees",
                        pipeline,
                        &[
                            BindingResource::Buffer(&uniform_buffer),
                            BindingResource::Buffer(&meteor_buffer),
                            BindingResource::Sampler(&sampler),
                            BindingResource::Texture(&forest_texture),
                        ],
                    );

                    render_pass.set_pipeline(&pipeline.pipeline);
                    render_pass.set_bind_group(0, bind_group, &[]);

                    render_pass.set_vertex_buffer(3, forest_points.slice(..));
                    tree.render(&mut render_pass, num_forest_points);
                }

                {
                    let pipeline = device.get_pipeline(
                        "house pipeline",
                        device.device.get_shader(
                            "shaders/compiled/house.vert.spv",
                            #[cfg(feature = "standalone")]
                            include_bytes!("../shaders/compiled/house.vert.spv"),
                            Default::default(),
                        ),
                        device.device.get_shader(
                            "shaders/compiled/tree.frag.spv",
                            #[cfg(feature = "standalone")]
                            include_bytes!("../shaders/compiled/tree.frag.spv"),
                            Default::default(),
                        ),
                        RenderPipelineDesc {
                            ..Default::default()
                        },
                        INSTANCED_FORMAT,
                    );

                    let bind_group = device.get_bind_group(
                        "house",
                        pipeline,
                        &[
                            BindingResource::Buffer(&uniform_buffer),
                            BindingResource::Buffer(&meteor_buffer),
                            BindingResource::Sampler(&sampler),
                            BindingResource::Texture(&house_texture),
                        ],
                    );

                    render_pass.set_pipeline(&pipeline.pipeline);
                    render_pass.set_bind_group(0, bind_group, &[]);

                    render_pass.set_vertex_buffer(3, house_points.slice(..));
                    house.render(&mut render_pass, num_house_points);
                }
            }

            drop(render_pass);

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("transmissiion render pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            {
                let formats = &[config.format];

                let device = device.with_formats(formats, Some(wgpu::TextureFormat::Depth32Float));

                let pipeline = device.get_pipeline(
                    "water pipeline",
                    device.device.get_shader(
                        "shaders/compiled/plane.vert.spv",
                        #[cfg(feature = "standalone")]
                        include_bytes!("../shaders/compiled/plane.vert.spv"),
                        Default::default(),
                    ),
                    device.device.get_shader(
                        "shaders/compiled/water.frag.spv",
                        #[cfg(feature = "standalone")]
                        include_bytes!("../shaders/compiled/water.frag.spv"),
                        BindGroupLayoutSettings {
                            allow_texture_filtering: false,
                        },
                    ),
                    RenderPipelineDesc {
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        ..Default::default()
                    },
                    BASIC_FORMAT,
                );

                let height_map = device
                    .device
                    .try_get_cached_texture("height map")
                    .expect("Failed to get cached texture.");

                let bind_group = device.get_bind_group(
                    "water",
                    pipeline,
                    &[
                        BindingResource::Buffer(&uniform_buffer),
                        BindingResource::Texture(opaque_texture),
                        BindingResource::Sampler(&sampler),
                        BindingResource::Texture(height_map),
                        BindingResource::Buffer(&meteor_buffer),
                    ],
                );

                render_pass.set_pipeline(&pipeline.pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);

                water.render(&mut render_pass, 1);

                {
                    let pipeline = device.get_pipeline(
                        "meteor outline pipeline",
                        device.device.get_shader(
                            "shaders/compiled/meteor_outline.vert.spv",
                            #[cfg(feature = "standalone")]
                            include_bytes!("../shaders/compiled/meteor_outline.vert.spv"),
                            Default::default(),
                        ),
                        device.device.get_shader(
                            "shaders/compiled/meteor_outline.frag.spv",
                            #[cfg(feature = "standalone")]
                            include_bytes!("../shaders/compiled/meteor_outline.frag.spv"),
                            Default::default(),
                        ),
                        RenderPipelineDesc {
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            ..Default::default()
                        },
                        BASIC_FORMAT,
                    );

                    let bind_group = device.get_bind_group(
                        "meteor outline",
                        pipeline,
                        &[
                            BindingResource::Buffer(&uniform_buffer),
                            BindingResource::Buffer(&meteor_buffer),
                            BindingResource::Sampler(&sampler),
                            BindingResource::Texture(&meteor_tex),
                        ],
                    );

                    render_pass.set_pipeline(&pipeline.pipeline);
                    render_pass.set_bind_group(0, bind_group, &[]);

                    meteor.render(&mut render_pass, 1);
                }

                let pipeline = device.get_pipeline(
                    "bounce sphere pipeline",
                    device.device.get_shader(
                        "shaders/compiled/bounce_sphere.vert.spv",
                        #[cfg(feature = "standalone")]
                        include_bytes!("../shaders/compiled/bounce_sphere.vert.spv"),
                        Default::default(),
                    ),
                    device.device.get_shader(
                        "shaders/compiled/bounce_sphere.frag.spv",
                        #[cfg(feature = "standalone")]
                        include_bytes!("../shaders/compiled/bounce_sphere.frag.spv"),
                        Default::default(),
                    ),
                    RenderPipelineDesc {
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),

                        ..Default::default()
                    },
                    BASIC_FORMAT,
                );

                let bind_group = device.get_bind_group(
                    "bounce sphere",
                    pipeline,
                    &[
                        BindingResource::Buffer(&uniform_buffer),
                        BindingResource::Buffer(&bounce_sphere_buffer),
                    ],
                );

                render_pass.set_pipeline(&pipeline.pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);

                bounce_sphere.render(&mut render_pass, 1);
            }

            drop(render_pass);

            {
                glyph_brush.queue(wgpu_glyph::Section {
                    screen_position: (16.0, 0.0),
                    bounds: (config.width as f32, config.height as f32),
                    text: vec![
                        wgpu_glyph::Text::new(&format!("Bounces: {}", state.bounces))
                            .with_color([0.0, 0.0, 0.0, 1.0])
                            .with_scale(24.0 * window.scale_factor() as f32),
                    ],
                    ..wgpu_glyph::Section::default()
                });

                if state.lost {
                    let text = if state.bounces >= 69 {
                        "Special Win\n特别胜利"
                    } else {
                        "Death\n死"
                    };

                    glyph_brush.queue(wgpu_glyph::Section {
                        screen_position: (config.width as f32 / 2.0, config.height as f32 / 2.0),
                        bounds: (config.width as f32, config.height as f32),
                        text: vec![wgpu_glyph::Text::new(text)
                            .with_color([0.75, 0.0, 0.0, 1.0])
                            .with_scale(96.0 * window.scale_factor() as f32)],
                        layout: wgpu_glyph::Layout::Wrap {
                            line_breaker: Default::default(),
                            h_align: wgpu_glyph::HorizontalAlign::Center,
                            v_align: wgpu_glyph::VerticalAlign::Center,
                        },
                    });
                }

                // Draw the text!
                glyph_brush
                    .draw_queued(
                        &device.inner,
                        &mut staging_belt,
                        &mut encoder,
                        &view,
                        config.width,
                        config.height,
                    )
                    .expect("Draw queued");

                staging_belt.finish();
            }

            queue.submit(std::iter::once(encoder.finish()));

            frame.present();

            device.flush();
        }
        _ => {}
    });
}

#[derive(Default)]
struct KeyboardState {
    left: bool,
    right: bool,
    up: bool,
    down: bool,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
#[repr(C)]
pub struct Uniforms {
    pub matrices: Mat4,
    pub player_position: Vec3,
    pub player_facing: f32,
    pub camera_position: Vec3,
    pub time: f32,
    pub window_size: Vec2,
    pub _padding: Vec2,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BounceSphereProps {
    pub position: Vec3,
    pub scale: f32,
}

pub struct MeteorProps {
    pub position: Vec3,
    pub velocity: Vec3,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct MeteorGpuProps {
    pub position: Vec3,
    pub _padding: u32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Similarity {
    pub translation: Vec3,
    pub scale: f32,
    pub rotation: Quat,
}

impl Similarity {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        scale: 1.0,
        rotation: Quat::IDENTITY,
    };

    pub fn as_mat4(self) -> Mat4 {
        Mat4::from_translation(self.translation)
            * Mat4::from_mat3(Mat3::from_quat(self.rotation))
            * Mat4::from_scale(Vec3::splat(self.scale))
    }
}

impl Mul<Similarity> for Similarity {
    type Output = Self;

    fn mul(self, child: Self) -> Self {
        Self {
            translation: self * child.translation,
            rotation: self.rotation * child.rotation,
            scale: self.scale * child.scale,
        }
    }
}

impl Mul<Vec3> for Similarity {
    type Output = Vec3;

    fn mul(self, vector: Vec3) -> Vec3 {
        self.translation + (self.scale * (self.rotation * vector))
    }
}

pub struct Orbit {
    pub pitch: f32,
    pub yaw: f32,
    distance: f32,
}

impl Orbit {
    fn from_vector(vector: Vec3) -> Self {
        let horizontal_length = (vector.x * vector.x + vector.z * vector.z).sqrt();
        let pitch = horizontal_length.atan2(vector.y);
        let distance = vector.length();
        Self {
            yaw: 0.0,
            pitch,
            distance,
        }
    }

    pub fn rotate(&mut self, delta: Vec2) {
        use std::f32::consts::PI;
        let speed = 0.15;
        self.yaw -= delta.x.to_radians() * speed;
        self.pitch = (self.pitch - delta.y.to_radians() * speed)
            .max(std::f32::EPSILON)
            .min(PI / 2.0);
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance + delta * 0.5).max(1.0).min(10.0);
    }

    fn as_vector(&self) -> Vec3 {
        let y = self.pitch.cos();
        let horizontal_amount = self.pitch.sin();
        let x = horizontal_amount * self.yaw.sin();
        let z = horizontal_amount * self.yaw.cos();
        Vec3::new(x, y, z) * self.distance
    }
}

fn short_angle_dist(from: f32, to: f32) -> f32 {
    let max_angle = std::f32::consts::PI * 2.0;
    let difference = n_mod_m(to - from, max_angle);
    n_mod_m(2.0 * difference, max_angle) - difference
}

fn n_mod_m<T: std::ops::Rem<Output = T> + std::ops::Add<Output = T> + Copy>(n: T, m: T) -> T {
    ((n % m) + m) % m
}

struct HeightMap {
    floats: Vec<f32>,
    height: u32,
    width: u32,
}

impl HeightMap {
    fn sample(&self, position: Vec2) -> f32 {
        let position = position * Vec2::new(self.width as f32, self.height as f32);

        let x = (position.x as u32).max(0).min(self.width - 1);
        let y = (position.y as u32).max(0).min(self.height - 1);

        let fractional_x = position.x % 1.0;
        let fractional_y = position.y % 1.0;

        fn lerp(x: f32, y: f32, f: f32) -> f32 {
            x * (1.0 - f) + y * f
        }

        let other_x = (x + 1).min(self.width - 1);
        let other_y = (y + 1).min(self.height - 1);

        let row_1 = lerp(self.fetch(x, y), self.fetch(other_x, y), fractional_x);
        let row_2 = lerp(
            self.fetch(x, other_y),
            self.fetch(other_x, other_y),
            fractional_x,
        );
        lerp(row_1, row_2, fractional_y)
    }

    fn fetch(&self, x: u32, y: u32) -> f32 {
        let pos = (y * self.width + x) as usize;

        self.floats[pos]
    }
}

fn main() {
    #[cfg(feature = "wasm")]
    wasm_bindgen_futures::spawn_local(run());
    #[cfg(not(feature = "wasm"))]
    pollster::block_on(run());
}

struct State {
    player_position: Vec2,
    player_speed: f32,
    player_facing: f32,
    time: f32,
    orbit: Orbit,
    bounce_sphere_props: BounceSphereProps,
    meteor_props: MeteorProps,
    kbd: KeyboardState,
    bounces: u32,
    lost: bool,
    animation_time: f32,
    animation_id: usize,
}

impl Default for State {
    fn default() -> Self {
        Self {
            player_position: Vec2::ZERO,
            player_speed: 0.0,
            player_facing: 0.0,
            time: 0.0,
            orbit: Orbit::from_vector(Vec3::new(0.0, 1.5, -3.5) * 2.5),
            bounce_sphere_props: BounceSphereProps {
                position: Vec3::ZERO,
                scale: 0.0,
            },
            meteor_props: MeteorProps {
                position: Vec3::Y * 100.0,
                velocity: Vec3::ZERO,
            },
            kbd: KeyboardState::default(),
            bounces: 0,
            lost: false,
            animation_time: 0.0,
            animation_id: 0,
        }
    }
}
