use glam::Mat4;
use glam::Quat;
use glam::Vec2;
use glam::Vec3;
use glam::Vec4;
use kiss_engine_wgpu::{
    BindGroupLayoutSettings, BindingResource, Device, RenderPipelineDesc, VertexBufferLayout,
};
use rand::Rng;

mod assets;

use assets::{load_image, Model};

use glow::HasContext;
use wgpu::util::DeviceExt;
use winit::event::VirtualKeyCode;

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

struct RawWindowHandle;

unsafe impl raw_window_handle::HasRawWindowHandle for RawWindowHandle {
    fn raw_window_handle(&self) -> raw_window_handle::RawWindowHandle {
        let mut web = raw_window_handle::WebHandle::empty();
        web.id = 666;

        raw_window_handle::RawWindowHandle::Web(web)
    }
}

async fn run() {
    #[cfg(feature = "wasm")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));

        let level: log::Level = wasm_web_helpers::parse_url_query_string_from_window("RUST_LOG")
            .and_then(|x| x.parse().ok())
            .unwrap_or(log::Level::Info);
        console_log::init_with_level(level).expect("could not initialize logger");
    }

    #[cfg(not(feature = "wasm"))]
    env_logger::init();

    use wasm_bindgen::JsCast;

    let vr_button = create_button("Start VR");
    let ar_button = create_button("Start AR");
    let inline_button = create_button("Start inline rendering");

    let start_vr_future = button_click_future(&vr_button);
    let start_ar_future = button_click_future(&ar_button);
    let start_inline_future = button_click_future(&inline_button);

    let canvas: web_sys::HtmlCanvasElement = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .unchecked_into();

    let body = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .body()
        .unwrap();

    canvas.set_attribute("data-raw-handle", "666").unwrap();

    body.append_child(&web_sys::Element::from(canvas.clone()))
        .unwrap();

    let video: web_sys::HtmlVideoElement = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("video")
        .unwrap()
        .unchecked_into();

    video.set_src("bbb.webm");
    video.set_autoplay(true);
    video.set_muted(true);
    video.set_loop(true);

    #[cfg(target_arch = "wasm32")]
    let webgl2_context = {
        let mut gl_attribs = std::collections::HashMap::new();
        gl_attribs.insert(String::from("xrCompatible"), true);
        let js_gl_attribs = wasm_bindgen::JsValue::from_serde(&gl_attribs).unwrap();

        canvas
            .get_context_with_context_options("webgl2", &js_gl_attribs)
            .unwrap()
            .unwrap()
            .dyn_into::<web_sys::WebGl2RenderingContext>()
            .unwrap()
    };

    let navigator: web_sys::Navigator = web_sys::window().unwrap().navigator();

    web_sys::console::log_1(&navigator);

    let xr = navigator.xr();

    web_sys::console::log_1(&xr);

    use futures::FutureExt;

    let mode = futures::select! {
        _ = Box::pin(start_vr_future.fuse()) => web_sys::XrSessionMode::ImmersiveVr,
        _ = Box::pin(start_ar_future.fuse()) => web_sys::XrSessionMode::ImmersiveAr,
        _ = Box::pin(start_inline_future.fuse()) => web_sys::XrSessionMode::Inline,
    };

    let reference_space_type = match mode {
        web_sys::XrSessionMode::Inline => web_sys::XrReferenceSpaceType::Viewer,
        _ => web_sys::XrReferenceSpaceType::Local,
    };

    let xr_session: web_sys::XrSession =
        wasm_bindgen_futures::JsFuture::from(xr.request_session(mode))
            .await
            .unwrap()
            .into();

    let xr_gl_layer =
        web_sys::XrWebGlLayer::new_with_web_gl2_rendering_context(&xr_session, &webgl2_context)
            .unwrap();

    let framebuffer_height = xr_gl_layer.framebuffer_height();
    let framebuffer_width = xr_gl_layer.framebuffer_width();

    let mut render_state_init = web_sys::XrRenderStateInit::new();
    render_state_init.base_layer(Some(&xr_gl_layer));
    xr_session.update_render_state_with_state(&render_state_init);

    let reference_space: web_sys::XrReferenceSpace = wasm_bindgen_futures::JsFuture::from(
        xr_session.request_reference_space(reference_space_type),
    )
    .await
    .unwrap()
    .into();

    let backend = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);
    let instance = wgpu::Instance::new(backend);
    let surface = unsafe { instance.create_surface(&RawWindowHandle) };
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
                limits: wgpu::Limits {
                    max_texture_dimension_1d: framebuffer_width.max(framebuffer_height).max(2048),
                    max_texture_dimension_2d: framebuffer_width.max(framebuffer_height).max(2048),
                    ..wgpu::Limits::downlevel_webgl2_defaults()
                },
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

    // Prepare glyph_brush
    let font = wgpu_glyph::ab_glyph::FontArc::try_from_slice(include_bytes!(
        "../VonwaonBitmap-16pxLite.ttf"
    ))
    .expect("Failed to load font");

    let mut glyph_brush = wgpu_glyph::GlyphBrushBuilder::using_font(font)
        .initial_cache_size((512, 512))
        .build(&device, wgpu::TextureFormat::Rgba8Unorm);

    let mut staging_belt = wgpu::util::StagingBelt::new(1024);

    let mut device = Device::new(device);

    let mut state = State::default();
    state.lost = true;

    let player_height_offset = Vec3::new(0.0, 1.5, 0.0);

    let uniform_buffer = device.create_resource(device.inner.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("uniform buffer"),
            contents: bytemuck::bytes_of(&Uniforms::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    ));

    let right_uniform_buffer = device.create_resource(device.inner.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("right uniform buffer"),
            contents: bytemuck::bytes_of(&Uniforms::default()),
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

        let bind_group = device.get_bind_group(("bake height map", 0), pipeline, &[]);

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

        fn cast_slice<T, F>(slice: &[T]) -> &[F] {
            unsafe {
                std::slice::from_raw_parts(
                    slice.as_ptr() as *const F,
                    (slice.len() * std::mem::size_of::<T>()) / std::mem::size_of::<F>(),
                )
            }
        }

        // We get bytemuck alignment issues for this.
        let vec4s: &[Vec4] = cast_slice(&bytes);

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
            size: (capsule.joint_indices_to_node_indices.len()
                * std::mem::size_of::<gltf_helpers::Similarity>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        }));

    wasm_bindgen_futures::JsFuture::from(video.play().unwrap())
        .await
        .unwrap();

    let mut gl_tex2 = None;

    unsafe {
        device.inner.as_hal::<wgpu_hal::gles::Api, _, _>(|device| {
            let device = device.unwrap();
            let gl = device.glow_context();

            let gl_tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(gl_tex));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );

            gl_tex2 = Some(gl_tex);
        });
    }

    let gl_tex = gl_tex2.unwrap();

    let external_texture = device.create_resource(
        device
            .inner
            .create_external_texture(Some("video"), Box::new(gl_tex)),
    );

    run_rendering_loop(&xr_session, move |time, frame| {
        let xr_session: web_sys::XrSession = frame.session();

        let pose = match frame.get_viewer_pose(&reference_space) {
            Some(pose) => pose,
            None => return,
        };

        let views: Vec<web_sys::XrView> = pose.views().iter().map(|view| view.into()).collect();

        struct Viewport {
            x: f32,
            y: f32,
            width: f32,
            height: f32,
        }

        let viewports: Vec<_> = views
            .iter()
            .map(|view| {
                let viewport = xr_gl_layer.get_viewport(view).unwrap();

                Viewport {
                    x: viewport.x() as f32,
                    y: viewport.y() as f32,
                    width: viewport.width() as f32,
                    height: viewport.height() as f32,
                }
            })
            .collect();

        let base_layer = xr_session.render_state().base_layer().unwrap();

        let width = base_layer.framebuffer_width() as f32;
        let height = base_layer.framebuffer_height() as f32;

        unsafe {
            let device = device.inner.clone();
            device.as_hal::<wgpu_hal::gles::Api, _, _>(|device| {
                let device = device.unwrap();

                let gl = device.glow_context();

                gl.bind_texture(glow::TEXTURE_2D, Some(gl_tex));
                let level = 0;
                let internal_format = glow::RGBA as i32;
                let src_format = glow::RGBA;
                let src_type = glow::UNSIGNED_BYTE;
                gl.tex_image_2d_with_html_video(
                    glow::TEXTURE_2D,
                    level,
                    internal_format,
                    src_format,
                    src_type,
                    &video,
                );
            });
        }

        // update
        {
            let delta_time = 1.0 / 60.0;

            state.update(delta_time, &heightmap, &capsule, &mut rng);

            capsule.animations[state.animation_id].animate(
                &mut player_joints,
                state.animation_time,
                &capsule.depth_first_nodes,
            );

            let joint_transforms: Vec<_> = player_joints
                .iter(
                    &capsule.joint_indices_to_node_indices,
                    &capsule.inverse_bind_transforms,
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

            let player_position_3d = state.player_position_3d(&heightmap);

            let parse_matrix = |vec| Mat4::from_cols_array(&<[f32; 16]>::try_from(vec).unwrap());

            let left_proj = parse_matrix(views[0].projection_matrix());

            let left_inv = parse_matrix(views[0].transform().inverse().matrix());

            queue.write_buffer(
                &uniform_buffer,
                0,
                bytemuck::bytes_of(&Uniforms {
                    matrices: { left_proj * (left_inv * Mat4::from_scale(Vec3::splat(0.1))) },
                    player_position: player_position_3d,
                    player_facing: state.player_facing,
                    camera_position: player_position_3d
                        + player_height_offset
                        + state.orbit.as_vector(),
                    time: state.time,
                    window_size: Vec2::new(width, height),
                    ..Default::default()
                }),
            );

            if let Some(right_view) = views.get(1) {
                let right_inv = parse_matrix(right_view.transform().inverse().matrix());
                let right_proj = parse_matrix(right_view.projection_matrix());

                queue.write_buffer(
                    &right_uniform_buffer,
                    0,
                    bytemuck::bytes_of(&Uniforms {
                        matrices: { right_proj * (right_inv * Mat4::from_scale(Vec3::splat(0.1))) },
                        player_position: player_position_3d,
                        player_facing: state.player_facing,
                        camera_position: player_position_3d
                            + player_height_offset
                            + state.orbit.as_vector(),
                        time: state.time,
                        window_size: Vec2::new(width, height),
                        ..Default::default()
                    }),
                );
            }
        }

        let framebuffer: web_sys::WebGlFramebuffer = base_layer.framebuffer();

        let opaque_texture = device.get_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            label: Some("opaque texture"),
        });

        let texture = unsafe {
            device.inner.create_texture_from_hal::<wgpu_hal::gles::Api>(
                wgpu_hal::gles::Texture {
                    inner: wgpu_hal::gles::TextureInner::Framebuffer {
                        inner: framebuffer.clone(),
                    },
                    mip_level_count: 1,
                    array_layer_count: 1,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    format_desc: wgpu_hal::gles::TextureFormatDesc {
                        internal: glow::RGBA,
                        external: glow::RGBA,
                        data_type: glow::UNSIGNED_BYTE,
                    },
                    copy_size: wgpu_hal::CopyExtent {
                        width: base_layer.framebuffer_width(),
                        height: base_layer.framebuffer_height(),
                        depth: 1,
                    },
                },
                &wgpu::TextureDescriptor {
                    label: Some("framebuffer (color)"),
                    size: wgpu::Extent3d {
                        width: base_layer.framebuffer_width(),
                        height: base_layer.framebuffer_height(),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                },
            )
        };

        let depth = unsafe {
            device.inner.create_texture_from_hal::<wgpu_hal::gles::Api>(
                wgpu_hal::gles::Texture {
                    inner: wgpu_hal::gles::TextureInner::Framebuffer { inner: framebuffer },
                    mip_level_count: 1,
                    array_layer_count: 1,
                    format: wgpu::TextureFormat::Depth32Float,
                    format_desc: wgpu_hal::gles::TextureFormatDesc {
                        internal: glow::RGBA,
                        external: glow::RGBA,
                        data_type: glow::UNSIGNED_BYTE,
                    },
                    copy_size: wgpu_hal::CopyExtent {
                        width: base_layer.framebuffer_width(),
                        height: base_layer.framebuffer_height(),
                        depth: 1,
                    },
                },
                &wgpu::TextureDescriptor {
                    label: Some("framebuffer (depth)"),
                    size: wgpu::Extent3d {
                        width: base_layer.framebuffer_width(),
                        height: base_layer.framebuffer_height(),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                },
            )
        };

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device
            .inner
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("command encoder"),
            });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main render pass"),
            color_attachments: &[
                wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if mode == web_sys::XrSessionMode::ImmersiveAr {
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
                                a: 1.0,
                            })
                        },
                        store: true,
                    },
                },
                /*wgpu::RenderPassColorAttachment {
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
                },*/
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        });

        let uniform_buffer = |i| {
            BindingResource::Buffer(if i == 0 {
                &uniform_buffer
            } else {
                &right_uniform_buffer
            })
        };

        {
            let formats = &[
                wgpu::TextureFormat::Rgba8Unorm,
                //wgpu::TextureFormat::Rgba8Unorm,
            ];

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
                    BindGroupLayoutSettings {
                        external_texture_slots: &[4],
                        ..Default::default()
                    },
                ),
                RenderPipelineDesc {
                    primitive: wgpu::PrimitiveState {
                        // as we're flipping things in the shaders.
                        cull_mode: Some(wgpu::Face::Front),
                        ..Default::default()
                    },
                    depth_compare: wgpu::CompareFunction::Less,
                    ..Default::default()
                },
                BASIC_FORMAT,
            );

            render_pass.set_pipeline(&pipeline.pipeline);

            for (i, viewport) in viewports.iter().enumerate() {
                render_pass.set_viewport(
                    viewport.x,
                    viewport.y,
                    viewport.width,
                    viewport.height,
                    0.0,
                    1.0,
                );
                let bind_group = device.get_bind_group(
                    ("plane pipeline", i),
                    pipeline,
                    &[
                        uniform_buffer(i),
                        BindingResource::Buffer(&meteor_buffer),
                        BindingResource::Sampler(&sampler),
                        BindingResource::Sampler(&linear_sampler),
                        //BindingResource::Texture(&grass_texture),
                        BindingResource::ExternalTexture(&external_texture),
                        BindingResource::Texture(&sand_texture),
                        BindingResource::Texture(&rock_texture),
                        BindingResource::Texture(&forest_texture),
                        BindingResource::Texture(&forest_map_tex),
                    ],
                );

                render_pass.set_bind_group(0, bind_group, &[]);
                plane.render(&mut render_pass, 1);
            }

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
                    primitive: wgpu::PrimitiveState {
                        // as we're flipping things in the shaders.
                        cull_mode: Some(wgpu::Face::Front),
                        ..Default::default()
                    },
                    depth_compare: wgpu::CompareFunction::Less,
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

            render_pass.set_pipeline(&pipeline.pipeline);

            render_pass.set_vertex_buffer(3, capsule.joints.slice(..));
            render_pass.set_vertex_buffer(4, capsule.weights.slice(..));

            for (i, viewport) in viewports.iter().enumerate() {
                render_pass.set_viewport(
                    viewport.x,
                    viewport.y,
                    viewport.width,
                    viewport.height,
                    0.0,
                    1.0,
                );

                let bind_group = device.get_bind_group(
                    ("capsule", i),
                    pipeline,
                    &[
                        uniform_buffer(i),
                        BindingResource::Buffer(&meteor_buffer),
                        BindingResource::Sampler(&sampler),
                        BindingResource::Texture(&giant_tex),
                        BindingResource::Buffer(&joint_transforms_buffer),
                    ],
                );

                render_pass.set_bind_group(0, bind_group, &[]);

                capsule.render(&mut render_pass, 1);
            }

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
                    primitive: wgpu::PrimitiveState {
                        // as we're flipping things in the shaders.
                        cull_mode: Some(wgpu::Face::Front),
                        ..Default::default()
                    },
                    depth_compare: wgpu::CompareFunction::Less,
                    ..Default::default()
                },
                BASIC_FORMAT,
            );

            render_pass.set_pipeline(&pipeline.pipeline);

            for (i, viewport) in viewports.iter().enumerate() {
                render_pass.set_viewport(
                    viewport.x,
                    viewport.y,
                    viewport.width,
                    viewport.height,
                    0.0,
                    1.0,
                );

                let bind_group = device.get_bind_group(
                    ("meteor", i),
                    pipeline,
                    &[
                        uniform_buffer(i),
                        BindingResource::Buffer(&meteor_buffer),
                        BindingResource::Sampler(&sampler),
                        BindingResource::Texture(&meteor_tex),
                    ],
                );
                render_pass.set_bind_group(0, bind_group, &[]);

                meteor.render(&mut render_pass, 1);
            }

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
                        primitive: wgpu::PrimitiveState {
                            // as we're flipping things in the shaders.
                            cull_mode: Some(wgpu::Face::Front),
                            ..Default::default()
                        },
                        depth_compare: wgpu::CompareFunction::Less,
                        ..Default::default()
                    },
                    INSTANCED_FORMAT,
                );

                render_pass.set_pipeline(&pipeline.pipeline);

                render_pass.set_vertex_buffer(3, forest_points.slice(..));

                for (i, viewport) in viewports.iter().enumerate() {
                    render_pass.set_viewport(
                        viewport.x,
                        viewport.y,
                        viewport.width,
                        viewport.height,
                        0.0,
                        1.0,
                    );

                    let bind_group = device.get_bind_group(
                        ("trees", i),
                        pipeline,
                        &[
                            uniform_buffer(i),
                            BindingResource::Buffer(&meteor_buffer),
                            BindingResource::Sampler(&sampler),
                            BindingResource::Texture(&forest_texture),
                        ],
                    );
                    render_pass.set_bind_group(0, bind_group, &[]);

                    tree.render(&mut render_pass, num_forest_points);
                }
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
                        primitive: wgpu::PrimitiveState {
                            // as we're flipping things in the shaders.
                            cull_mode: Some(wgpu::Face::Front),
                            ..Default::default()
                        },
                        depth_compare: wgpu::CompareFunction::Less,
                        ..Default::default()
                    },
                    INSTANCED_FORMAT,
                );

                render_pass.set_pipeline(&pipeline.pipeline);

                render_pass.set_vertex_buffer(3, house_points.slice(..));

                for (i, viewport) in viewports.iter().enumerate() {
                    render_pass.set_viewport(
                        viewport.x,
                        viewport.y,
                        viewport.width,
                        viewport.height,
                        0.0,
                        1.0,
                    );

                    let bind_group = device.get_bind_group(
                        ("house", i),
                        pipeline,
                        &[
                            uniform_buffer(i),
                            BindingResource::Buffer(&meteor_buffer),
                            BindingResource::Sampler(&sampler),
                            BindingResource::Texture(&house_texture),
                        ],
                    );
                    render_pass.set_bind_group(0, bind_group, &[]);

                    house.render(&mut render_pass, num_house_points);
                }
            }
        }

        drop(render_pass);

        /*encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All
            },
            wgpu::ImageCopyTexture {
                texture: &opaque_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All
            },
            wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            }
        );*/

        /*let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("transmisiion render pass"),
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });

        {
            let formats = &[wgpu::TextureFormat::Rgba8Unorm];

            let device = device.with_formats(formats, None);

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
                        ..Default::default()
                    },
                ),
                RenderPipelineDesc {
                    primitive: wgpu::PrimitiveState {
                        // as we're flipping things in the shaders.
                        cull_mode: Some(wgpu::Face::Front),
                        ..Default::default()
                    },
                    depth_compare: wgpu::CompareFunction::Less,
                    ..Default::default()
                },
                BASIC_FORMAT,
            );

            let height_map = device
                .device
                .try_get_cached_texture("height map")
                .expect("Failed to get cached texture.");

            for (i, viewport) in viewports.iter().enumerate() {
                render_pass.set_viewport(
                    viewport.x,
                    viewport.y,
                    viewport.width,
                    viewport.height,
                    0.0,
                    1.0,
                );

                let bind_group = device.get_bind_group(
                    ("water", i),
                    pipeline,
                    &[
                        uniform_buffer(i),
                        BindingResource::Texture(opaque_texture),
                        BindingResource::Sampler(&sampler),
                        BindingResource::Texture(height_map),
                        BindingResource::Buffer(&meteor_buffer),
                    ],
                );

                render_pass.set_pipeline(&pipeline.pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);

                water.render(&mut render_pass, 1);
            }

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
                        primitive: wgpu::PrimitiveState {
                            // as we're flipping things in the shaders.
                            cull_mode: Some(wgpu::Face::Front),
                            ..Default::default()
                        },
                        depth_compare: wgpu::CompareFunction::Less,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        ..Default::default()
                    },
                    BASIC_FORMAT,
                );

                render_pass.set_pipeline(&pipeline.pipeline);

                for (i, viewport) in viewports.iter().enumerate() {
                    render_pass.set_viewport(
                        viewport.x,
                        viewport.y,
                        viewport.width,
                        viewport.height,
                        0.0,
                        1.0,
                    );

                    let bind_group = device.get_bind_group(
                        ("meteor outline", i),
                        pipeline,
                        &[
                            uniform_buffer(i),
                            BindingResource::Buffer(&meteor_buffer),
                            BindingResource::Sampler(&sampler),
                            BindingResource::Texture(&meteor_tex),
                        ],
                    );

                    render_pass.set_bind_group(0, bind_group, &[]);

                    meteor.render(&mut render_pass, 1);
                }
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
                    primitive: wgpu::PrimitiveState {
                        // as we're flipping things in the shaders.
                        cull_mode: Some(wgpu::Face::Front),
                        ..Default::default()
                    },
                    depth_compare: wgpu::CompareFunction::Less,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    ..Default::default()
                },
                BASIC_FORMAT,
            );

            render_pass.set_pipeline(&pipeline.pipeline);

            for (i, viewport) in viewports.iter().enumerate() {
                render_pass.set_viewport(
                    viewport.x,
                    viewport.y,
                    viewport.width,
                    viewport.height,
                    0.0,
                    1.0,
                );

                let bind_group = device.get_bind_group(
                    ("bounce sphere", i),
                    pipeline,
                    &[
                        uniform_buffer(i),
                        BindingResource::Buffer(&bounce_sphere_buffer),
                    ],
                );

                render_pass.set_bind_group(0, bind_group, &[]);

                bounce_sphere.render(&mut render_pass, 1);
            }
        }

        drop(render_pass);*/

        /*{
            let scale_factor = || -> f64 { 1.0 };

            let draw_text = |glyph_brush: &mut wgpu_glyph::GlyphBrush<()>| {
                glyph_brush.queue(wgpu_glyph::Section {
                    screen_position: (16.0, 0.0),
                    bounds: (width as f32 / 2.0, height as f32),
                    text: vec![
                        wgpu_glyph::Text::new(&format!("Bounces: {}", state.bounces))
                            .with_color([0.0, 0.0, 0.0, 1.0])
                            .with_scale(24.0 * scale_factor() as f32),
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
                        screen_position: (width as f32 / 4.0, height as f32 / 2.0),
                        bounds: (width as f32 / 2.0, height as f32),
                        text: vec![wgpu_glyph::Text::new(text)
                            .with_color([0.75, 0.0, 0.0, 1.0])
                            .with_scale(96.0 * scale_factor() as f32)],
                        layout: wgpu_glyph::Layout::Wrap {
                            line_breaker: Default::default(),
                            h_align: wgpu_glyph::HorizontalAlign::Center,
                            v_align: wgpu_glyph::VerticalAlign::Center,
                        },
                    });
                }
            };

            {
                draw_text(&mut glyph_brush);

                // Draw the text!
                glyph_brush
                    .draw_queued_with_transform(
                        &device.inner,
                        &mut staging_belt,
                        &mut encoder,
                        &view,
                        (Mat4::from_cols_array(&wgpu_glyph::orthographic_projection(
                            width as u32,
                            height as u32,
                        )) * Mat4::from_translation(Vec3::new(
                            width as f32 / 2.0,
                            0.0,
                            0.0,
                        )))
                        .to_cols_array(),
                    )
                    .expect("Draw queued");

                draw_text(&mut glyph_brush);

                glyph_brush
                    .draw_queued_with_transform(
                        &device.inner,
                        &mut staging_belt,
                        &mut encoder,
                        &view,
                        Mat4::from_cols_array(&wgpu_glyph::orthographic_projection(
                            width as u32,
                            height as u32,
                        ))
                        .to_cols_array(),
                    )
                    .expect("Draw queued");

                staging_belt.finish();
            }
        }*/

        queue.submit(std::iter::once(encoder.finish()));

        device.flush();
    })
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
pub(crate) struct Uniforms {
    pub(crate) matrices: Mat4,
    pub(crate) player_position: Vec3,
    pub(crate) player_facing: f32,
    pub(crate) camera_position: Vec3,
    pub(crate) time: f32,
    pub(crate) window_size: Vec2,
    pub(crate) _padding: Vec2,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(crate) struct BounceSphereProps {
    pub(crate) position: Vec3,
    pub(crate) scale: f32,
}

pub(crate) struct MeteorProps {
    pub(crate) position: Vec3,
    pub(crate) velocity: Vec3,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(crate) struct MeteorGpuProps {
    pub(crate) position: Vec3,
    pub(crate) _padding: u32,
}

pub(crate) struct Orbit {
    pub(crate) pitch: f32,
    pub(crate) yaw: f32,
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
                position: Vec3::new(5.0, 5.0, 5.0),
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

impl State {
    fn player_position_3d(&self, heightmap: &HeightMap) -> Vec3 {
        Vec3::new(
            self.player_position.x,
            heightmap.sample(self.player_position / 80.0 + 0.5),
            self.player_position.y,
        )
    }

    fn update(
        &mut self,
        delta_time: f32,
        heightmap: &HeightMap,
        player_model: &Model,
        rng: &mut rand::rngs::OsRng,
    ) {
        self.time += delta_time;

        let forwards = self.kbd.up as i32 - self.kbd.down as i32;
        let right = self.kbd.right as i32 - self.kbd.left as i32;

        if (forwards, right) != (0, 0) {
            if self.animation_id == 0 {
                self.animation_id = 2;
                self.animation_time = 0.0;
            }

            let new_player_facing = self.orbit.yaw
                + match (forwards, right) {
                    (0, 1) => -90.0_f32.to_radians(),
                    (0, -1) => 90.0_f32.to_radians(),
                    (1, 1) => -45.0_f32.to_radians(),
                    (1, -1) => 45.0_f32.to_radians(),

                    (-1, -1) => 135.0_f32.to_radians(),
                    (-1, 1) => -135.0_f32.to_radians(),

                    (-1, 0) => 180.0_f32.to_radians(),

                    _ => 0.0,
                };

            self.player_facing += short_angle_dist(self.player_facing, new_player_facing) * 0.5;

            self.orbit.yaw += short_angle_dist(self.orbit.yaw, self.player_facing) * 0.015;

            self.player_speed = (self.player_speed + delta_time).min(1.0);
        } else {
            self.player_speed *= 0.9;

            if self.animation_id == 2 {
                self.animation_id = 0;
                self.animation_time = 0.0;
            }
        }

        let movement = Quat::from_rotation_y(self.player_facing)
            * Vec3::new(0.0, 0.0, -delta_time * 10.0 * self.player_speed);

        self.player_position.x += movement.x;
        self.player_position.y += movement.z;

        self.player_position.x = self.player_position.x.max(-40.0).min(40.0);
        self.player_position.y = self.player_position.y.max(-40.0).min(40.0);

        self.bounce_sphere_props.scale += self.bounce_sphere_props.scale * delta_time * 0.5;

        if self.bounce_sphere_props.scale > 1.5 {
            self.bounce_sphere_props.scale = 0.0;
        }

        self.meteor_props.velocity.y -= delta_time * 5.0;

        if (self.meteor_props.position.x + self.meteor_props.velocity.x * delta_time).abs() > 40.0 {
            self.meteor_props.velocity.x *= -1.0;
        }

        if (self.meteor_props.position.z + self.meteor_props.velocity.z * delta_time).abs() > 40.0 {
            self.meteor_props.velocity.z *= -1.0;
        }

        //self.meteor_props.position += self.meteor_props.velocity * delta_time;

        if self.bounce_sphere_props.scale > 0.0
            && self
                .meteor_props
                .position
                .distance(self.bounce_sphere_props.position)
                < (self.bounce_sphere_props.scale + 1.0)
        {
            // This code can run move than once per bounce so we just do this hack lol
            if self.meteor_props.velocity.y <= 0.0 {
                self.bounces += 1;
            }

            self.meteor_props.velocity.y = self.meteor_props.velocity.y.abs();

            let rotation = rng.gen_range(0.0..=std::f32::consts::TAU);
            let velocity = rng.gen_range(1.0..=5.0);

            self.meteor_props.velocity.x = velocity * rotation.cos();
            self.meteor_props.velocity.z = velocity * rotation.sin();
        } else if self.meteor_props.position.y < 0.0 {
            self.lost = true;

            self.meteor_props.position.y -= 100.0;
        }

        if self.animation_id == 1
            && self.animation_time > 0.25
            && self.animation_time < 0.75
            && self.bounce_sphere_props.scale == 0.0
        {
            self.bounce_sphere_props.scale = 1.0;
            self.bounce_sphere_props.position = self.player_position_3d(heightmap) + Vec3::Y * 4.0;
        }

        self.animation_time += delta_time;

        while self.animation_time > player_model.animations[self.animation_id].total_time() {
            self.animation_time -= player_model.animations[self.animation_id].total_time();

            if self.animation_id == 1 {
                self.animation_id = 0;
            }
        }
    }

    fn handle_key(&mut self, key: VirtualKeyCode, pressed: bool) {
        match key {
            VirtualKeyCode::W | VirtualKeyCode::Up => self.kbd.up = pressed,
            VirtualKeyCode::A | VirtualKeyCode::Left => self.kbd.left = pressed,
            VirtualKeyCode::S | VirtualKeyCode::Down => self.kbd.down = pressed,
            VirtualKeyCode::D | VirtualKeyCode::Right => self.kbd.right = pressed,
            VirtualKeyCode::C if pressed => {
                self.orbit.yaw = self.player_facing;
            }
            VirtualKeyCode::Space if pressed && self.animation_id != 1 => {
                self.animation_id = 1;
                self.animation_time = 0.0;
            }
            VirtualKeyCode::R if pressed && self.lost => {
                *self = Self::default();
            }
            _ => {}
        }
    }
}

fn create_button(text: &str) -> web_sys::HtmlButtonElement {
    use wasm_bindgen::JsCast;

    let button: web_sys::HtmlButtonElement = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("button")
        .unwrap()
        .unchecked_into();

    button.set_inner_text(text);

    let body = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .body()
        .unwrap();

    body.append_child(&web_sys::Element::from(button.clone()))
        .unwrap();

    button
}

async fn button_click_future(button: &web_sys::HtmlButtonElement) {
    wasm_bindgen_futures::JsFuture::from(js_sys::Promise::new(&mut |resolve, _reject| {
        button.set_onclick(Some(&resolve))
    }))
    .await
    .unwrap();
}

fn run_rendering_loop<F: FnMut(f64, web_sys::XrFrame) + 'static>(
    session: &web_sys::XrSession,
    mut func: F,
) {
    use std::cell::RefCell;
    use std::rc::Rc;
    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::JsCast;

    fn request_animation_frame(
        session: &web_sys::XrSession,
        f: &Closure<dyn FnMut(f64, web_sys::XrFrame)>,
    ) -> u32 {
        // This turns the Closure into a js_sys::Function
        // See https://rustwasm.github.io/wasm-bindgen/api/wasm_bindgen/closure/struct.Closure.html#casting-a-closure-to-a-js_sysfunction
        session.request_animation_frame(f.as_ref().unchecked_ref())
    }

    // Wierd hacky closure stuff that I don't understand. Taken from wasm-bindgen.
    // TODO: link source.
    let closure = Rc::new(RefCell::new(None));
    let closure_clone = closure.clone();

    *closure.borrow_mut() = Some(Closure::wrap(Box::new(
        move |time: f64, frame: web_sys::XrFrame| {
            let session = frame.session();

            request_animation_frame(&session, closure_clone.borrow().as_ref().unwrap());

            func(time, frame);
        },
    )
        as Box<dyn FnMut(f64, web_sys::XrFrame)>));

    request_animation_frame(&session, closure.borrow().as_ref().unwrap());
}
