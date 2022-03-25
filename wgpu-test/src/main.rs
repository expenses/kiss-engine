use winit::{
    event::{self, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};


#[cfg(target_arch = "wasm32")]
use wasm_web_helpers::parse_url_query_string_from_window;

use kiss_engine_wgpu::RenderPipelineDesc;

async fn run() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }

    #[cfg(target_arch = "wasm32")]
    {
        let level: log::Level = parse_url_query_string_from_window("RUST_LOG")
            .map(|x| x.parse().ok())
            .flatten()
            .unwrap_or(log::Level::Info);
        console_log::init_with_level(level).expect("could not initialize logger");
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    }

    log::info!("Starting");

    let event_loop = EventLoop::new();
    let builder = winit::window::WindowBuilder::new();

    let window = builder.build(&event_loop).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        // On wasm, append the canvas to the document body
        wasm_web_helpers::append_canvas(&window);
    }

    let backend = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);

    let instance = wgpu::Instance::new(backend);
    let size = window.inner_size();
    let surface = unsafe { instance.create_surface(&window) };

    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }).await
    .expect("No suitable GPU adapters found on the system!");

    let adapter_info = adapter.get_info();
    log::info!(
        "Using {} with the {:?} backend",
        adapter_info.name, adapter_info.backend
    );

    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("device"),
            features: Default::default(),
            limits: Default::default(),
        },
        None,
    ).await
    .expect("Unable to find a suitable GPU adapter!");

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface.get_preferred_format(&adapter).unwrap(),
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
    };
    surface.configure(&device, &config);

    let mut device = kiss_engine_wgpu::Device::new(device);

    let uniform_buffer = device.create_resource(device.inner.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform buffer"),
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));

    let mut time: f32 = 0.0;

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
            log::info!("Resizing to {:?}", size);
            config.width = size.width.max(1);
            config.height = size.height.max(1);
            surface.configure(&device.inner, &config);
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
            _ => {}
        },
        event::Event::MainEventsCleared => {
            time += 1.0 / 60.0;

            queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&time));

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

            let depth_buffer = device.get_texture(&wgpu::TextureDescriptor {
                label: Some("depth buffer"),
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
            });

            let mut encoder =
                device
                    .inner
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("command encoder"),
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
                    view: depth_buffer,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            {
                let device = device.with_formats(config.format, Some(wgpu::TextureFormat::Depth32Float));

                let pipeline = device.get_pipeline(
                    "blit pipeline",
                    device.device.get_shader(
                        "vert.hlsl.spv",
                        #[cfg(target_arch = "wasm32")]
                        include_bytes!("../vert.hlsl.spv")
                    ),
                    device.device.get_shader(
                        "frag.hlsl.spv",
                        #[cfg(target_arch = "wasm32")]
                        include_bytes!("../frag.hlsl.spv")
                    ),
                    RenderPipelineDesc {
                        depth_compare: wgpu::CompareFunction::Always,
                        ..Default::default()
                    },
                    &[],
                );

                let bind_group = device.get_bind_group("blit bind group", pipeline, &[kiss_engine_wgpu::BindingResource::Buffer(&uniform_buffer)]);

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

fn main() {
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(run());
    #[cfg(not(target_arch = "wasm32"))]
    pollster::block_on(run());
}
