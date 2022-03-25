mod level;

use kiss_engine_wgpu::{
    BindingResource, Device, RenderPipelineDesc, Resource, VertexBufferLayout, DeviceWithFormats,
};

use wgpu::util::DeviceExt;
use winit::{
    event::{self, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

pub const Z_NEAR: f32 = 0.01;
pub const Z_FAR: f32 = 50_000.0;

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

static mut SM64: Option<libsm64::Sm64> = None;

fn main() {
    env_logger::init();

    use libsm64::*;

    unsafe {
        SM64 = Some(Sm64::new(&include_bytes!("../mario.z64")[..]).unwrap());
    }

    let mario = unsafe {
        let sm64 = SM64.as_ref().unwrap();
        sm64.load_level_geometry(level::POINTS);
        let mario = sm64.create_mario(0, 1000, 0).unwrap();

        mario
    };

    let event_loop = EventLoop::new();
    let builder = winit::window::WindowBuilder::new();

    let window = builder.build(&event_loop).unwrap();

    let backend = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);

    let instance = wgpu::Instance::new(backend);
    let size = window.inner_size();
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: Some(&surface),
    }))
    .expect("No suitable GPU adapters found on the system!");

    let adapter_info = adapter.get_info();
    println!(
        "Using {} with the {:?} backend",
        adapter_info.name, adapter_info.backend
    );

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("device"),
            features: Default::default(),
            limits: Default::default(),
        },
        None,
    ))
    .expect("Unable to find a suitable GPU adapter!");

    fn vertex_to_vec(vertex: libsm64::Point3<i16>) -> glam::Vec3 {
        glam::Vec3::new(vertex.x as f32, vertex.y as f32, vertex.z as f32) / 50.0
    }

    let triangles: Vec<Triangle> = level::POINTS
        .iter()
        .map(|triangle| {
            let a = vertex_to_vec(triangle.vertices.0);
            let b = vertex_to_vec(triangle.vertices.1);
            let c = vertex_to_vec(triangle.vertices.2);

            Triangle {
                a,
                b,
                c,
                normal: {
                    let edge_b_a = b - a;
                    let edge_c_a = c - a;
                    let crossed_normal = edge_b_a.cross(edge_c_a);

                    crossed_normal.normalize()
                },
            }
        })
        .collect();

    let buffer: Vec<glam::Vec3> = triangles
        .iter()
        .flat_map(|triangle| [triangle.a, triangle.b, triangle.c])
        .collect();

    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("geo buffer"),
        contents: bytemuck::cast_slice(&buffer),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let normal_buffer: Vec<glam::Vec3> = triangles
        .iter()
        .flat_map(|triangle| [triangle.normal, triangle.normal, triangle.normal])
        .collect();

    let normal_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("geo normal buffer"),
        contents: bytemuck::cast_slice(&normal_buffer),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let grass = {
        let densities: Vec<f32> = triangles
            .iter()
            .map(|triangle| {
                let area = {
                    // https://en.wikipedia.org/wiki/Heron%27s_formula#Formulation

                    let edge_a = triangle.a.distance(triangle.b);
                    let edge_b = triangle.b.distance(triangle.c);
                    let edge_c = triangle.c.distance(triangle.a);

                    let semi_perimeter = (edge_a + edge_b + edge_c) / 2.0;

                    ((semi_perimeter - edge_a)
                        * (semi_perimeter - edge_b)
                        * (semi_perimeter - edge_c)
                        * semi_perimeter)
                        .sqrt()
                };

                let mut slope_modifier = triangle.normal.dot(glam::Vec3::Y).max(0.0);

                if slope_modifier < 0.5 {
                    slope_modifier = 0.0;
                }

                area * slope_modifier
            })
            .collect();

        let max_density = densities
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        use rand::Rng;

        let mut rng = rand::thread_rng();

        let mut currently_allocated = 0.0;
        let max = 200_000.0;

        let mut allocations = vec![0; triangles.len()];

        while currently_allocated < max {
            let i = rng.gen_range(0..triangles.len());

            allocations[i] += 1;

            currently_allocated += densities[i] / max_density;
        }

        dbg!(allocations.iter().sum::<usize>());

        allocations
            .iter_mut()
            .enumerate()
            .for_each(|(i, allocation)| {
                *allocation = (*allocation as f32 * (densities[i] / max_density)) as usize;
            });

        dbg!(allocations.iter().sum::<usize>());

        let points: Vec<glam::Vec3> = allocations
            .iter()
            .enumerate()
            .flat_map(|(i, num_allocations)| std::iter::repeat(i).take(*num_allocations))
            .map(|i| {
                let tri = &triangles[i];

                let mut x: f32 = rng.gen_range(0.0..1.0);
                let mut y: f32 = rng.gen_range(0.0..1.0);

                if x + y > 1.0 {
                    x = 1.0 - x;
                    y = 1.0 - y;
                }

                let z = 1.0 - x - y;

                tri.a * x + tri.b * y + tri.c * z
            })
            .collect();

        points
    };

    let num_points = grass.len() as u32;

    let grass = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grass buffer"),
        contents: bytemuck::cast_slice(&grass),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let mut mario_buffers = MarioBuffers {
        position: ResizingBuffer::new(&device, "mario position buffer"),
        uv: ResizingBuffer::new(&device, "mario uv buffer"),
        normal: ResizingBuffer::new(&device, "mario normal buffer"),
        colour: ResizingBuffer::new(&device, "mario colour buffer"),
    };

    let swapchain_format = surface.get_preferred_format(&adapter).unwrap();

    let camera = dolly::rig::CameraRig::builder()
        .with(dolly::drivers::Position::new(glam::Vec3::new(
            0.0,
            1000.0 / 50.0,
            0.0,
        )))
        .with(dolly::drivers::YawPitch::new().pitch_degrees(15.0))
        .with(dolly::drivers::Arm::new(glam::Vec3::Z * 10.0))
        .build();

    let mut perspective_matrix = perspective_matrix_reversed(size.width, size.height);

    let view_matrix = glam::Mat4::look_at_rh(
        camera.final_transform.position,
        camera.final_transform.position + camera.final_transform.forward(),
        camera.final_transform.up(),
    );

    let mut wireframe = false;

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface.get_preferred_format(&adapter).unwrap(),
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
    };
    surface.configure(&device, &config);

    let mut device = Device::new(device);

    let texture = {
        let texture = unsafe { SM64.as_ref() }.unwrap().texture();

        let texture = device.inner.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("mario texture"),
                size: wgpu::Extent3d {
                    width: texture.width,
                    height: texture.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
            },
            texture.data,
        );

        device.create_resource(texture.create_view(&wgpu::TextureViewDescriptor::default()))
    };

    let uniform_buffer = device.create_resource(device.inner.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("uniform buffer"),
            contents: bytemuck::bytes_of(&{ perspective_matrix * view_matrix }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    ));

    let sampler = device.create_resource(
        device
            .inner
            .create_sampler(&wgpu::SamplerDescriptor::default()),
    );

    let state = std::sync::Arc::new(parking_lot::Mutex::new(State {
        mario,
        mario_state: Default::default(),
        camera,
    }));

    let gamepad_state = std::sync::Arc::new(parking_lot::Mutex::new(GamepadState::default()));

    std::thread::spawn({
        let state = state.clone();
        let gamepad_state = gamepad_state.clone();
        move || mario_loop(state, gamepad_state)
    });
    std::thread::spawn({
        let gamepad_state = gamepad_state.clone();
        move || gamepad_input_loop(gamepad_state)
    });

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
                        virtual_keycode: Some(event::VirtualKeyCode::W),
                        state: event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                wireframe = !wireframe;
            }
            _ => {}
        },
        event::Event::MainEventsCleared => {
            let mut state = state.lock();

            let delta_time = 1.0 / 60.0;

            let mario_position = state.mario_state.position;

            {
                let gamepad_state = gamepad_state.lock();
                let yaw = -gamepad_state.right_stick_x;
                let pitch = gamepad_state.right_stick_y;

                state
                    .camera
                    .driver_mut::<dolly::drivers::YawPitch>()
                    .rotate_yaw_pitch(yaw * 2.0, pitch * 2.0);

                state
                    .camera
                    .driver_mut::<dolly::drivers::Position>()
                    .position =
                    glam::Vec3::new(mario_position.x, mario_position.y, mario_position.z) / 50.0;

                state.camera.update(delta_time);
            }

            let view_matrix = glam::Mat4::look_at_rh(
                state.camera.final_transform.position,
                state.camera.final_transform.position + state.camera.final_transform.forward(),
                state.camera.final_transform.up(),
            );

            queue.write_buffer(
                &uniform_buffer,
                0,
                bytemuck::bytes_of(&{ perspective_matrix * view_matrix }),
            );

            let mario_geom = state.mario.geometry();

            unsafe fn unsafe_cast_slice<T>(slice: &[T]) -> &[u8] {
                std::slice::from_raw_parts(
                    slice.as_ptr() as *const u8,
                    slice.len() * std::mem::size_of::<T>(),
                )
            }

            unsafe {
                mario_buffers.position.update(
                    &device.inner,
                    &queue,
                    unsafe_cast_slice(mario_geom.positions()),
                );
                mario_buffers
                    .uv
                    .update(&device.inner, &queue, unsafe_cast_slice(mario_geom.uvs()));
                mario_buffers.colour.update(
                    &device.inner,
                    &queue,
                    unsafe_cast_slice(mario_geom.colors()),
                );
                mario_buffers.normal.update(
                    &device.inner,
                    &queue,
                    unsafe_cast_slice(mario_geom.normals()),
                );
            }

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

            let hdr_texture = device.get_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d {
                    width: config.width,
                    height: config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                label: Some("hdr texture"),
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
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                label: Some("depth texture"),
            });

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main render pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &hdr_texture,
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
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            {
                let device = device.with_formats(wgpu::TextureFormat::Rgba16Float, Some(wgpu::TextureFormat::Depth32Float));

                render_mario(
                    &device,
                    &mut render_pass,
                    &mario_buffers,
                    &uniform_buffer,
                    &sampler,
                    &texture,
                    wireframe,
                );

                render_grass(&device, &mut render_pass, &uniform_buffer, &grass, num_points);

                render_world(&device, &mut render_pass, &uniform_buffer, &buffer, &normal_buffer);
            }

            drop(render_pass);

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit render pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
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
                }],
                depth_stencil_attachment: None,
            });

            {
                let device = device.with_formats(swapchain_format, None);

                let pipeline = device.get_pipeline(
                    "blit pipeline",
                    device.device.get_shader("shaders/full_screen_tri.vert.spv"),
                    device.device.get_shader("shaders/blit.frag.spv"),
                    RenderPipelineDesc::default(),
                    &[],
                );
                let bind_group = device.get_bind_group("blit bind group", pipeline, &[
                    BindingResource::Sampler(&sampler),
                    BindingResource::Texture(&hdr_texture),
                ]);

                render_pass.set_pipeline(&pipeline.pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);
                render_pass.draw(0..3, 0..1);
            }

            drop(render_pass);

            queue.submit(std::iter::once(encoder.finish()));

            frame.present();

            device.flush();
        }
        _ => {}
    });
}

struct MarioBuffers {
    position: ResizingBuffer,
    uv: ResizingBuffer,
    normal: ResizingBuffer,
    colour: ResizingBuffer,
}

struct State {
    mario: libsm64::Mario<'static>,
    camera: dolly::rig::CameraRig,
    mario_state: libsm64::MarioState,
}

fn mario_loop(
    state: std::sync::Arc<parking_lot::Mutex<State>>,
    gamepad_state: std::sync::Arc<parking_lot::Mutex<GamepadState>>,
) {
    loop {
        let start = std::time::Instant::now();

        let mut state = state.lock();
        let gamepad_state = gamepad_state.lock();

        let mut input = libsm64::MarioInput {
            cam_look_x: state.mario_state.position.x
                - state.camera.final_transform.position.x * 50.0,
            cam_look_z: state.mario_state.position.z
                - state.camera.final_transform.position.z * 50.0,
            stick_x: gamepad_state.left_stick_x,
            stick_y: -gamepad_state.left_stick_y,
            button_a: gamepad_state.button_south,
            button_b: gamepad_state.button_west,
            button_z: gamepad_state.button_north,
        };

        drop(gamepad_state);

        // Deadzoning

        if input.stick_x.abs() < 5e-5 {
            input.stick_x = 0.0;
        }

        if input.stick_y.abs() < 5e-5 {
            input.stick_y = 0.0;
        }

        state.mario_state = state.mario.tick(input);

        drop(state);

        let now = std::time::Instant::now();

        let update_time = now - start;

        std::thread::sleep(std::time::Duration::from_secs_f64(1.0 / 30.0) - update_time);
    }
}

#[derive(Default)]
struct GamepadState {
    left_stick_x: f32,
    left_stick_y: f32,
    right_stick_x: f32,
    right_stick_y: f32,
    button_west: bool,
    button_south: bool,
    button_north: bool,
}

fn gamepad_input_loop(state: std::sync::Arc<parking_lot::Mutex<GamepadState>>) {
    let mut gilrs = gilrs::Gilrs::new().unwrap();

    let mut active_gamepad = None;

    loop {
        while let Some(gilrs::ev::Event { id, .. }) = gilrs.next_event() {
            active_gamepad = Some(id);
        }

        if let Some(gamepad) = active_gamepad.map(|id| gilrs.gamepad(id)) {
            let temp_state = GamepadState {
                left_stick_x: gamepad.value(gilrs::ev::Axis::LeftStickX),
                left_stick_y: gamepad.value(gilrs::ev::Axis::LeftStickY),
                right_stick_x: gamepad.value(gilrs::ev::Axis::RightStickX),
                right_stick_y: gamepad.value(gilrs::ev::Axis::RightStickY),
                button_west: gamepad.is_pressed(gilrs::ev::Button::West),
                button_south: gamepad.is_pressed(gilrs::ev::Button::South),
                button_north: gamepad.is_pressed(gilrs::ev::Button::North),
            };

            *state.lock() = temp_state;
        }

        std::thread::sleep(std::time::Duration::from_secs_f64(1.0 / 60.0));
    }
}

struct ResizingBuffer {
    buffer: wgpu::Buffer,
    capacity: u64,
    len: u32,
    label: &'static str,
}

impl ResizingBuffer {
    fn new(device: &wgpu::Device, label: &'static str) -> Self {
        Self {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 0,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            capacity: 0,
            label,
            len: 0,
        }
    }

    fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) {
        if bytes.len() as u64 > self.capacity {
            self.buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(self.label),
                contents: bytes,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            self.capacity = bytes.len() as u64;
        } else {
            queue.write_buffer(&self.buffer, 0, bytes);
        }

        self.len = bytes.len() as u32;
    }
}

fn render_mario<'a>(
    device: &DeviceWithFormats<'a>,
    render_pass: &mut wgpu::RenderPass<'a>,
    mario_buffers: &'a MarioBuffers,
    uniform_buffer: &Resource<wgpu::Buffer>,
    sampler: &Resource<wgpu::Sampler>,
    texture: &Resource<wgpu::TextureView>,
    nega_mario: bool,
) {
    let buffer_layout = &[
        VertexBufferLayout {
            location: 0,
            format: wgpu::VertexFormat::Float32x3,
            step_mode: wgpu::VertexStepMode::Vertex,
        },
        VertexBufferLayout {
            location: 1,
            format: wgpu::VertexFormat::Float32x2,
            step_mode: wgpu::VertexStepMode::Vertex,
        },
        VertexBufferLayout {
            location: 2,
            format: wgpu::VertexFormat::Float32x3,
            step_mode: wgpu::VertexStepMode::Vertex,
        },
        VertexBufferLayout {
            location: 3,
            format: wgpu::VertexFormat::Float32x3,
            step_mode: wgpu::VertexStepMode::Vertex,
        },
    ];

    let bindings = &[
        BindingResource::Buffer(uniform_buffer),
        BindingResource::Sampler(sampler),
        BindingResource::Texture(texture),
    ];

    if nega_mario {
        let pipeline = device.get_pipeline(
            "nega mario pipeline",
            device.device.get_shader("shaders/mario.vert.spv"),
            device.device.get_shader("shaders/nega_mario.frag.spv"),
            RenderPipelineDesc::default(),
            buffer_layout,
        );

        let bind_group = device.get_bind_group("nega mario bind group", pipeline, bindings);

        render_pass.set_pipeline(&pipeline.pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
    } else {
        let pipeline = device.get_pipeline(
            "mario pipeline",
            device.device.get_shader("shaders/mario.vert.spv"),
            device.device.get_shader("shaders/mario.frag.spv"),
            RenderPipelineDesc::default(),
            buffer_layout,
        );

        let bind_group = device.get_bind_group("mario bind group", pipeline, bindings);

        render_pass.set_pipeline(&pipeline.pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
    }

    render_pass
        .set_vertex_buffer(0, mario_buffers.position.buffer.slice(..));
    render_pass
        .set_vertex_buffer(1, mario_buffers.uv.buffer.slice(..));
    render_pass
        .set_vertex_buffer(2, mario_buffers.colour.buffer.slice(..));
    render_pass
        .set_vertex_buffer(3, mario_buffers.normal.buffer.slice(..));
    render_pass.draw(
        0..mario_buffers.position.len / std::mem::size_of::<glam::Vec3>() as u32,
        0..1,
    );
}

fn render_world<'a>(
    device: &DeviceWithFormats<'a>,
    render_pass: &mut wgpu::RenderPass<'a>,
    uniform_buffer: &'a Resource<wgpu::Buffer>,
    buffer: &'a wgpu::Buffer,
    normal_buffer: &'a wgpu::Buffer,
) {
    let pipeline = device.get_pipeline(
        "world pipeline",
        device.device.get_shader("shaders/world.vert.spv"),
        device.device.get_shader("shaders/world.frag.spv"),
        RenderPipelineDesc::default(),
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
        ],
    );

    let bind_group = device.get_bind_group("world bind group", pipeline,         &[BindingResource::Buffer(uniform_buffer)],
);


    render_pass.set_pipeline(&pipeline.pipeline);
    render_pass.set_bind_group(0, bind_group, &[]);
    render_pass.set_vertex_buffer(0, buffer.slice(..));
    render_pass.set_vertex_buffer(1, normal_buffer.slice(..));
    render_pass
        .draw(0..level::POINTS.len() as u32 * 3, 0..1);
}

fn render_grass<'a>(
    device: &DeviceWithFormats<'a>,
    render_pass: &mut wgpu::RenderPass<'a>,
    uniform_buffer: &'a Resource<wgpu::Buffer>,
    buffer: &'a wgpu::Buffer,
    num_points: u32,
) {
    let pipeline = device.get_pipeline(
        "grass pipeline",
        device.device.get_shader("shaders/point.vert.spv"),
        device.device.get_shader("shaders/flat_colour.frag.spv"),
        RenderPipelineDesc {
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            ..Default::default()
        },
        &[VertexBufferLayout {
            location: 0,
            format: wgpu::VertexFormat::Float32x3,
            step_mode: wgpu::VertexStepMode::Instance,
        }],
    );

    let bind_group = device.get_bind_group("grass bind group",  pipeline,       &[BindingResource::Buffer(uniform_buffer)]);

    render_pass.set_pipeline(&pipeline.pipeline);
    render_pass.set_bind_group(0, bind_group, &[]);
    render_pass.set_vertex_buffer(0, buffer.slice(..));
    render_pass.draw(0..2, 0..num_points);
}

struct Triangle {
    a: glam::Vec3,
    b: glam::Vec3,
    c: glam::Vec3,
    normal: glam::Vec3,
}
