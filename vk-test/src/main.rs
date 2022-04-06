use ash::extensions::ext;
use ash::extensions::khr;
use ash::vk;
use std::ffi::CStr;
use std::os::raw::c_char;
use winit::{
    event::{self, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
};

#[path = "../../its-a-me/src/level.rs"]
mod level;

use kiss_engine_vk::pipeline_resources::{GraphicsPipelineSettings, VertexBufferLayout};

use kiss_engine_vk::binding_resources::{BindingResource, Buffer};
use kiss_engine_vk::device::Device;
use kiss_engine_vk::primitives::{vulkan_debug_utils_callback, CStrList, Swapchain};

pub const Z_NEAR: f32 = 0.01;
pub const Z_FAR: f32 = 500.0;

fn perspective_matrix_reversed(width: u32, height: u32) -> glam::Mat4 {
    let aspect_ratio = width as f32 / height as f32;
    let vertical_fov = 59.0_f32.to_radians();

    let focal_length = 1.0 / (vertical_fov / 2.0).tan();

    let a = Z_NEAR / (Z_FAR - Z_NEAR);
    let b = Z_FAR * a;

    glam::Mat4::from_cols(
        glam::Vec4::new(focal_length / aspect_ratio, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, -focal_length, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, a, -1.0),
        glam::Vec4::new(0.0, 0.0, b, 0.0),
    )
}

fn main() {
    env_logger::init();

    let mut event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .build(&event_loop)
        .unwrap();

    let entry = unsafe { ash::Entry::load() }.unwrap();

    let app_info = vk::ApplicationInfo::builder()
        .application_name(c_str_macro::c_str!("KISS Graph"))
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_1);

    let instance_extensions = {
        let mut instance_extensions = ash_window::enumerate_required_extensions(&window).unwrap().to_vec();
        instance_extensions.extend(&[
            ext::DebugUtils::name().as_ptr(),
            khr::GetPhysicalDeviceProperties2::name().as_ptr(),
        ]);
        instance_extensions
    };

    let enabled_layers = CStrList::new(vec![CStr::from_bytes_with_nul(
        b"VK_LAYER_KHRONOS_validation\0",
    )
    .unwrap()]);

    let mut debug_messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_utils_callback));

    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_extension_names(&instance_extensions)
                .enabled_layer_names(enabled_layers.pointers())
                .push_next(&mut debug_messenger_info),
            None,
        )
    }
    .unwrap();

    let debug_utils_ext = ext::DebugUtils::new(&entry, &instance);

    let surface_ext = khr::Surface::new(&entry, &instance);

    let surface = unsafe { ash_window::create_surface(&entry, &instance, &window, None) }.unwrap();

    let required_extensions =
        CStrList::new(vec![khr::Swapchain::name(), khr::DynamicRendering::name()]);

    let (physical_device, queue_family, surface_format) = {
        let physical_devices = unsafe { instance.enumerate_physical_devices() }.unwrap();

        log::info!("Found {} device(s)", physical_devices.len(),);

        let mut acceptable_devices = Vec::new();

        unsafe {
            for physical_device in physical_devices {
                let properties = instance.get_physical_device_properties(physical_device);

                log::info!(
                    "Found device: {:?}",
                    cstr_from_array(&properties.device_name)
                );

                log::info!(
                    "Api version: {}.{}.{}",
                    vk::api_version_major(properties.api_version),
                    vk::api_version_minor(properties.api_version),
                    vk::api_version_patch(properties.api_version)
                );

                let queue_family = instance
                    .get_physical_device_queue_family_properties(physical_device)
                    .into_iter()
                    .enumerate()
                    .position(|(i, queue_family_properties)| {
                        queue_family_properties
                            .queue_flags
                            .contains(vk::QueueFlags::GRAPHICS)
                            && surface_ext
                                .get_physical_device_surface_support(
                                    physical_device,
                                    i as u32,
                                    surface,
                                )
                                .unwrap()
                    })
                    .map(|queue_family| queue_family as u32)
                    .unwrap();

                let surface_formats = surface_ext
                    .get_physical_device_surface_formats(physical_device, surface)
                    .unwrap();

                let surface_format = surface_formats
                    .iter()
                    .find(|surface_format| {
                        surface_format.format == vk::Format::B8G8R8A8_SRGB
                            && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                    })
                    .or_else(|| surface_formats.get(0))
                    .cloned()
                    .unwrap();

                let supported_device_extensions = instance
                    .enumerate_device_extension_properties(physical_device)
                    .unwrap();

                let mut has_required_extensions = true;

                for required_extension in &required_extensions.list {
                    let device_has_extension =
                        supported_device_extensions.iter().any(|extension| {
                            &cstr_from_array(&extension.extension_name) == required_extension
                        });

                    log::info!("* {:?}: {}", required_extension, tick(device_has_extension));

                    has_required_extensions &= device_has_extension;
                }

                if !has_required_extensions {
                    break;
                }

                acceptable_devices.push((physical_device, queue_family, surface_format));
            }
        }

        acceptable_devices[0]
    };

    let device = {
        let queue_info = [*vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family)
            .queue_priorities(&[1.0])];

        let device_features = vk::PhysicalDeviceFeatures::builder();

        let mut dynamic_rendering_features =
            vk::PhysicalDeviceDynamicRenderingFeaturesKHR::builder().dynamic_rendering(true);

        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_info)
            .enabled_features(&device_features)
            .enabled_extension_names(required_extensions.pointers())
            .enabled_layer_names(enabled_layers.pointers())
            .push_next(&mut dynamic_rendering_features);

        unsafe { instance.create_device(physical_device, &device_info, None) }.unwrap()
    };

    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    let command_pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family),
            None,
        )
    }
    .unwrap();

    let command_buffers = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::builder()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .unwrap();

    let command_buffer = command_buffers[0];

    let allocator =
        gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings {
                log_leaks_on_shutdown: true,
                ..Default::default()
            },
            buffer_device_address: false,
        })
        .unwrap();

    let surface_caps =
        unsafe { surface_ext.get_physical_device_surface_capabilities(physical_device, surface) }
            .unwrap();

    let mut image_count = surface_caps.min_image_count.max(3);
    if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
        image_count = surface_caps.max_image_count;
    }

    log::info!("Using {} swapchain images at a time.", image_count);

    let swapchain_ext = khr::Swapchain::new(&instance, &device);

    let size = window.inner_size();

    let mut extent = vk::Extent2D {
        width: size.width,
        height: size.height,
    };

    let mut swapchain_info = *vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let dynamic_rendering_ext = khr::DynamicRendering::new(&instance, &device);

    let mut device = Device::new(device, allocator, debug_utils_ext, swapchain_ext);

    let mut swapchain = Swapchain::new(&device, &swapchain_info);

    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    let create_named_semaphore = |name| {
        let semaphore = unsafe { device.inner.create_semaphore(&semaphore_info, None) }.unwrap();

        device.set_object_name(semaphore, name);

        semaphore
    };

    let present_semaphore = create_named_semaphore("present semaphore");
    let render_semaphore = create_named_semaphore("render semaphore");
    let render_fence = unsafe { device.inner.create_fence(&fence_info, None).unwrap() };

    let camera: dolly::rig::CameraRig = dolly::rig::CameraRig::builder()
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

    let mut uniform_buffer = device.create_resource(Buffer::new_from_bytes(
        &device,
        bytemuck::bytes_of(&{ perspective_matrix * view_matrix }),
        "uniform buffer",
        vk::BufferUsageFlags::UNIFORM_BUFFER,
    ));

    fn vertex_to_vec(vertex: libsm64::Point3<i16>) -> glam::Vec3 {
        glam::Vec3::new(vertex.x as f32, vertex.y as f32, vertex.z as f32) / 50.0
    }

    struct Triangle {
        a: glam::Vec3,
        b: glam::Vec3,
        c: glam::Vec3,
        normal: glam::Vec3,
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

    let buffer = device.create_resource(Buffer::new_from_bytes(
        &device,
        bytemuck::cast_slice(&buffer),
        "geo buffer",
        vk::BufferUsageFlags::VERTEX_BUFFER,
    ));

    let normal_buffer: Vec<glam::Vec3> = triangles
        .iter()
        .flat_map(|triangle| [triangle.normal, triangle.normal, triangle.normal])
        .collect();

    let normal_buffer = device.create_resource(Buffer::new_from_bytes(
        &device,
        bytemuck::cast_slice(&normal_buffer),
        "geo normal buffer",
        vk::BufferUsageFlags::VERTEX_BUFFER,
    ));

    event_loop.run_return(|event, _, control_flow| match event {
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

            extent.width = size.width;
            extent.height = size.height;
            swapchain_info.image_extent = extent;
            swapchain_info.old_swapchain = swapchain.inner;

            unsafe {
                device.inner.queue_wait_idle(queue).unwrap();
            }

            swapchain = Swapchain::new(&device, &swapchain_info);

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
            _ => {}
        },
        event::Event::MainEventsCleared => {
            let view_matrix = glam::Mat4::look_at_rh(
                camera.final_transform.position,
                camera.final_transform.position + camera.final_transform.forward(),
                camera.final_transform.up(),
            );

            uniform_buffer
                .inner
                .write_mapped(bytemuck::bytes_of(&{ perspective_matrix * view_matrix }), 0);

            /*
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
                mario_buffers.color.update(
                    &device.inner,
                    &queue,
                    unsafe_cast_slice(mario_geom.colors()),
                );
                mario_buffers.normal.update(
                    &device.inner,
                    &queue,
                    unsafe_cast_slice(mario_geom.normals()),
                );
            }*/

            window.request_redraw();
        }
        event::Event::RedrawRequested(_) => unsafe {
            device
                .inner
                .wait_for_fences(&[render_fence], true, u64::MAX)
                .unwrap();

            device.inner.reset_fences(&[render_fence]).unwrap();

            device
                .inner
                .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())
                .unwrap();

            let swapchain_image_index = match device.swapchain.acquire_next_image(
                swapchain.inner,
                u64::MAX,
                present_semaphore,
                vk::Fence::null(),
            ) {
                Ok((swapchain_image_index, _suboptimal)) => swapchain_image_index,
                Err(error) => {
                    log::warn!("Next frame error: {:?}", error);
                    return;
                }
            };

            {
                device
                    .inner
                    .begin_command_buffer(
                        command_buffer,
                        &vk::CommandBufferBeginInfo::builder()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .unwrap();

                let subres = vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1);

                let depth_buffer = device.get_image(
                    "depth buffer",
                    extent,
                    vk::Format::D32_SFLOAT,
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                );

                let depth_subres = vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .level_count(1)
                    .layer_count(1);

                vk_sync::cmd::pipeline_barrier(
                    &device.inner,
                    command_buffer,
                    None,
                    &[],
                    &[
                        vk_sync::ImageBarrier {
                            previous_accesses: &[vk_sync::AccessType::Present],
                            next_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
                            discard_contents: true,
                            image: swapchain.images[swapchain_image_index as usize],
                            range: *subres,
                            ..Default::default()
                        },
                        vk_sync::ImageBarrier {
                            next_accesses: &[vk_sync::AccessType::DepthStencilAttachmentWrite],
                            discard_contents: true,
                            image: depth_buffer.image,
                            range: *depth_subres,
                            ..Default::default()
                        },
                    ],
                );

                let attachment = vk::RenderingAttachmentInfoKHR::builder()
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image_view(swapchain.views[swapchain_image_index as usize])
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.1, 0.2, 0.3, 1.0],
                        },
                    });

                let depth_attachment_clear = vk::RenderingAttachmentInfoKHR::builder()
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .image_view(depth_buffer.view)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 0.0,
                            stencil: 0,
                        },
                    });

                let depth_attachment_load = vk::RenderingAttachmentInfoKHR::builder()
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .image_view(depth_buffer.view)
                    .load_op(vk::AttachmentLoadOp::LOAD);

                let area = vk::Rect2D {
                    offset: Default::default(),
                    extent,
                };

                let viewport = *vk::Viewport::builder()
                    .x(0.0)
                    .y(0.0)
                    .width(extent.width as f32)
                    .height(extent.height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0);

                device.inner.cmd_set_scissor(command_buffer, 0, &[area]);
                device
                    .inner
                    .cmd_set_viewport(command_buffer, 0, &[viewport]);

                {
                    dynamic_rendering_ext.cmd_begin_rendering(
                        command_buffer,
                        &vk::RenderingInfoKHR::builder()
                            .layer_count(1)
                            .render_area(area)
                            .depth_attachment(&depth_attachment_clear),
                    );

                    {
                        {
                            let world_pipeline = device.get_graphics_pipeline(
                                "world depth re pass pipeline",
                                device.get_shader("../its-a-me/shaders/world.vert.spv"),
                                None,
                                &Default::default(),
                                &[],
                                &[
                                    VertexBufferLayout {
                                        location: 0,
                                        format: vk::Format::R32G32B32_SFLOAT,
                                        input_rate: vk::VertexInputRate::VERTEX,
                                    },
                                    VertexBufferLayout {
                                        location: 1,
                                        format: vk::Format::R32G32B32_SFLOAT,
                                        input_rate: vk::VertexInputRate::VERTEX,
                                    },
                                ],
                            );

                            device.inner.cmd_bind_pipeline(
                                command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                world_pipeline.pipeline,
                            );

                            let ds = device.get_descriptor_set(
                                "world depth pre pass ds",
                                world_pipeline,
                                &[BindingResource::Buffer(&uniform_buffer)],
                            );

                            device.inner.cmd_bind_descriptor_sets(
                                command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                world_pipeline.pipeline_layout,
                                0,
                                &[ds.inner],
                                &[],
                            );

                            device.inner.cmd_bind_vertex_buffers(
                                command_buffer,
                                0,
                                &[buffer.buffer, normal_buffer.buffer],
                                &[0, 0],
                            );

                            device.inner.cmd_draw(
                                command_buffer,
                                level::POINTS.len() as u32 * 3,
                                1,
                                0,
                                0,
                            );
                        }
                    }

                    dynamic_rendering_ext.cmd_end_rendering(command_buffer);
                }

                vk_sync::cmd::pipeline_barrier(
                    &device.inner,
                    command_buffer,
                    None,
                    &[],
                    &[vk_sync::ImageBarrier {
                        previous_accesses: &[vk_sync::AccessType::DepthStencilAttachmentWrite],
                        next_accesses: &[vk_sync::AccessType::DepthStencilAttachmentRead],
                        image: depth_buffer.image,
                        range: *depth_subres,
                        ..Default::default()
                    }],
                );

                dynamic_rendering_ext.cmd_begin_rendering(
                    command_buffer,
                    &vk::RenderingInfoKHR::builder()
                        // NEEDS TO BE THERE.
                        .layer_count(1)
                        // NEEDS TO BE THERE.
                        .render_area(area)
                        .color_attachments(&[*attachment])
                        .depth_attachment(&depth_attachment_load),
                );

                {
                    let world_pipeline = device.get_graphics_pipeline(
                        "world pipeline",
                        device.get_shader("../its-a-me/shaders/world.vert.spv"),
                        Some(device.get_shader("../its-a-me/shaders/world.frag.spv")),
                        &GraphicsPipelineSettings {
                            depth_write_enable: false,
                            depth_compare_op: vk::CompareOp::EQUAL,
                            ..Default::default()
                        },
                        &[vk::Format::B8G8R8A8_SRGB],
                        &[
                            VertexBufferLayout {
                                location: 0,
                                format: vk::Format::R32G32B32_SFLOAT,
                                input_rate: vk::VertexInputRate::VERTEX,
                            },
                            VertexBufferLayout {
                                location: 1,
                                format: vk::Format::R32G32B32_SFLOAT,
                                input_rate: vk::VertexInputRate::VERTEX,
                            },
                        ],
                    );

                    device.inner.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        world_pipeline.pipeline,
                    );

                    let ds = device.get_descriptor_set(
                        "world ds",
                        world_pipeline,
                        &[BindingResource::Buffer(&uniform_buffer)],
                    );

                    device.inner.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        world_pipeline.pipeline_layout,
                        0,
                        &[ds.inner],
                        &[],
                    );

                    device.inner.cmd_bind_vertex_buffers(
                        command_buffer,
                        0,
                        &[buffer.buffer, normal_buffer.buffer],
                        &[0, 0],
                    );

                    device
                        .inner
                        .cmd_draw(command_buffer, level::POINTS.len() as u32 * 3, 1, 0, 0);
                }

                dynamic_rendering_ext.cmd_end_rendering(command_buffer);

                vk_sync::cmd::pipeline_barrier(
                    &device.inner,
                    command_buffer,
                    None,
                    &[],
                    &[vk_sync::ImageBarrier {
                        previous_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
                        next_accesses: &[vk_sync::AccessType::Present],
                        image: swapchain.images[swapchain_image_index as usize],
                        range: *subres,
                        ..Default::default()
                    }],
                );

                device.inner.end_command_buffer(command_buffer).unwrap();
            }

            {
                device
                    .inner
                    .queue_submit(
                        queue,
                        &[*vk::SubmitInfo::builder()
                            .wait_semaphores(&[present_semaphore])
                            .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
                            .command_buffers(&[command_buffer])
                            .signal_semaphores(&[render_semaphore])],
                        render_fence,
                    )
                    .unwrap();
            }

            {
                device
                    .swapchain
                    .queue_present(
                        queue,
                        &vk::PresentInfoKHR::builder()
                            .wait_semaphores(&[render_semaphore])
                            .swapchains(&[swapchain.inner])
                            .image_indices(&[swapchain_image_index]),
                    )
                    .unwrap();
            }

            device.flush();
        },
        _ => {}
    });

    unsafe {
        device.inner.queue_wait_idle(queue).unwrap();

        device.inner.destroy_semaphore(present_semaphore, None);
        device.inner.destroy_semaphore(render_semaphore, None);
        device.inner.destroy_fence(render_fence, None);
        device.inner.destroy_command_pool(command_pool, None);
    }

    drop(uniform_buffer);
    drop(buffer);
    drop(normal_buffer);
    drop(swapchain);

    let raw_device = device.inner.clone();

    drop(device);

    unsafe {
        raw_device.destroy_device(None);

        surface_ext.destroy_surface(surface, None);

        instance.destroy_instance(None);
    }
}

fn tick(supported: bool) -> &'static str {
    if supported {
        "✔️"
    } else {
        "❌"
    }
}

unsafe fn cstr_from_array(array: &[c_char]) -> &CStr {
    CStr::from_ptr(array.as_ptr())
}
