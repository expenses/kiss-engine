use glam::Mat3;
use glam::Mat4;
use glam::Quat;
use glam::Vec3;
use kiss_engine_wgpu::{
    BindingResource, Device, DeviceWithFormats, RenderPipelineDesc, Resource, VertexBufferLayout,
};
use std::ops::*;

use wgpu::util::DeviceExt;
use winit::{
    event::{self, VirtualKeyCode, WindowEvent},
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

fn main() {
    env_logger::init();

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

    let plane = Model::new(include_bytes!("../plane.glb"), &device, "plane").unwrap();
    let capsule = Model::new(include_bytes!("../capsule.glb"), &device, "capsule").unwrap();

    let mut perspective_matrix = perspective_matrix_reversed(size.width, size.height);

    let size = window.inner_size();

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface.get_preferred_format(&adapter).unwrap(),
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
    };
    surface.configure(&device, &config);

    let mut device = Device::new(device);

    let mut camera: dolly::rig::CameraRig = dolly::rig::CameraRig::builder()
        .with(dolly::drivers::Position::new(glam::Vec3::new(
            0.0, 0.0, 0.0,
        )))
        .with(dolly::drivers::YawPitch::new().pitch_degrees(-15.0))
        .with(dolly::drivers::Arm::new(glam::Vec3::Z * 10.0))
        .build();

    let view_matrix = glam::Mat4::look_at_rh(
        camera.final_transform.position,
        camera.final_transform.position + camera.final_transform.forward(),
        camera.final_transform.up(),
    );

    let uniform_buffer = device.create_resource(device.inner.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("uniform buffer"),
            contents: bytemuck::bytes_of(&{ perspective_matrix * view_matrix }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    ));

    let mut kbd = KeyboardState::default();

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
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state == event::ElementState::Pressed;

                match key {
                    VirtualKeyCode::W => kbd.up = pressed,
                    VirtualKeyCode::A => kbd.left = pressed,
                    VirtualKeyCode::S => kbd.down = pressed,
                    VirtualKeyCode::D => kbd.right = pressed,
                    _ => {}
                }
            }
            _ => {}
        },
        event::Event::MainEventsCleared => {
            let yaw = kbd.right as i32 as f32 - kbd.left as i32 as f32;
            let pitch = kbd.down as i32 as f32 - kbd.up as i32 as f32;

            camera
                .driver_mut::<dolly::drivers::YawPitch>()
                .rotate_yaw_pitch(yaw * 2.0, pitch * 2.0);

            camera.driver_mut::<dolly::drivers::Position>().position =
                glam::Vec3::new(0.0, 0.0, 0.0);

            let delta_time = 1.0 / 60.0;

            camera.update(delta_time);

            let view_matrix = glam::Mat4::look_at_rh(
                camera.final_transform.position,
                camera.final_transform.position + camera.final_transform.forward(),
                camera.final_transform.up(),
            );

            queue.write_buffer(
                &uniform_buffer,
                0,
                bytemuck::bytes_of(&{ perspective_matrix * view_matrix }),
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
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                label: Some("depth texture"),
            });

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main render pass"),
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
                let device =
                    device.with_formats(config.format, Some(wgpu::TextureFormat::Depth32Float));

                let pipeline = device.get_pipeline(
                    "plane pipeline",
                    device.device.get_shader("shaders/plane.vert.spv"),
                    device.device.get_shader("shaders/plane.frag.spv"),
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
                    ],
                );

                let bind_group = device.get_bind_group(
                    "plane bind group",
                    pipeline,
                    &[kiss_engine_wgpu::BindingResource::Buffer(&uniform_buffer)],
                );

                render_pass.set_pipeline(&pipeline.pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);

                render_pass.set_vertex_buffer(0, capsule.positions.slice(..));
                render_pass.set_vertex_buffer(1, capsule.normals.slice(..));
                render_pass.set_index_buffer(capsule.indices.slice(..), wgpu::IndexFormat::Uint32);

                render_pass.draw_indexed(0..capsule.num_indices, 0, 0..1);

                render_pass.set_vertex_buffer(0, plane.positions.slice(..));
                render_pass.set_vertex_buffer(1, plane.normals.slice(..));
                render_pass.set_index_buffer(plane.indices.slice(..), wgpu::IndexFormat::Uint32);

                render_pass.draw_indexed(0..plane.num_indices, 0, 0..1);
            }

            drop(render_pass);

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

struct Model {
    positions: wgpu::Buffer,
    normals: wgpu::Buffer,
    indices: wgpu::Buffer,
    num_indices: u32,
}

impl Model {
    fn new(bytes: &[u8], device: &wgpu::Device, name: &str) -> anyhow::Result<Self> {
        let gltf = gltf::Gltf::from_slice(bytes)?;

        let buffer_blob = gltf.blob.as_ref().unwrap();

        let node_tree = NodeTree::new(gltf.nodes());

        let mut indices = Vec::new();
        let mut positions = Vec::new();
        let mut normals = Vec::new();

        for (node, mesh) in gltf
            .nodes()
            .filter_map(|node| node.mesh().map(|mesh| (node, mesh)))
        {
            let transform = node_tree.transform_of(node.index());

            for primitive in mesh.primitives() {
                let reader = primitive.reader(|buffer| {
                    assert_eq!(buffer.index(), 0);
                    Some(buffer_blob)
                });

                let read_indices = reader.read_indices().unwrap().into_u32();

                let num_existing_vertices = positions.len();

                let first_index = indices.len();

                indices.extend(read_indices.map(|index| index + num_existing_vertices as u32));

                positions.extend(
                    reader
                        .read_positions()
                        .unwrap()
                        .map(|pos| transform * Vec3::from(pos)),
                );

                normals.extend(
                    reader
                        .read_positions()
                        .unwrap()
                        .map(|normal| transform.rotation * Vec3::from(normal)),
                );
            }
        }

        Ok(Self {
            positions: (device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} positions buffer", name)),
                contents: bytemuck::cast_slice(&positions),
                usage: wgpu::BufferUsages::VERTEX,
            })),
            indices: (device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} indices buffer", name)),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            })),
            normals: (device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} normals buffer", name)),
                contents: bytemuck::cast_slice(&normals),
                usage: wgpu::BufferUsages::VERTEX,
            })),
            num_indices: indices.len() as u32,
        })
    }
}
pub struct NodeTree {
    inner: Vec<(Similarity, usize)>,
}

impl NodeTree {
    fn new(nodes: gltf::iter::Nodes) -> Self {
        let mut inner = vec![(Similarity::IDENTITY, usize::max_value()); nodes.clone().count()];

        for node in nodes {
            let (translation, rotation, scale) = node.transform().decomposed();

            assert!(
                (scale[0] - scale[1]).abs() <= std::f32::EPSILON * 10.0,
                "{:?}",
                scale
            );
            assert!(
                (scale[0] - scale[2]).abs() <= std::f32::EPSILON * 10.0,
                "{:?}",
                scale
            );

            inner[node.index()].0 = Similarity {
                translation: translation.into(),
                rotation: Quat::from_array(rotation),
                scale: scale[0],
            };
            for child in node.children() {
                inner[child.index()].1 = node.index();
            }
        }

        Self { inner }
    }

    pub fn transform_of(&self, mut index: usize) -> Similarity {
        let mut transform_sum = Similarity::IDENTITY;

        while index != usize::max_value() {
            let (transform, parent_index) = self.inner[index];
            transform_sum = transform * transform_sum;
            index = parent_index;
        }

        transform_sum
    }
}

#[derive(Clone, Copy)]
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
