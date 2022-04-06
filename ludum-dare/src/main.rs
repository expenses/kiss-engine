use ash::extensions::ext;
use ash::extensions::khr;
use ash::vk;
use glam::Mat4;
use glam::Quat;
use glam::Vec2;
use glam::Vec3;
use glam::Vec4;
/*
use kiss_engine_wgpu::{
    BindGroupLayoutSettings, BindingResource, Device, RenderPipelineDesc, VertexBufferLayout,
};*/
use kiss_engine_vk::binding_resources::BindingResource;
use kiss_engine_vk::cstr_from_array;
use kiss_engine_vk::primitives::{vulkan_debug_utils_callback, CStrList, Swapchain};
use kiss_engine_vk::VertexBufferLayout;
use rand::Rng;
use std::ffi::CStr;
use winit::platform::run_return::EventLoopExtRunReturn;

mod assets;

use assets::{load_image, Model};

use winit::{
    event::{self, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

pub(crate) const Z_NEAR: f32 = 0.01;
pub(crate) const Z_FAR: f32 = 50_000.0;

const BASIC_FORMAT: &[VertexBufferLayout] = &[
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
    VertexBufferLayout {
        location: 2,
        format: vk::Format::R32G32_SFLOAT,
        input_rate: vk::VertexInputRate::VERTEX,
    },
];

const INSTANCED_FORMAT: &[VertexBufferLayout] = &[
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
    VertexBufferLayout {
        location: 2,
        format: vk::Format::R32G32_SFLOAT,
        input_rate: vk::VertexInputRate::VERTEX,
    },
    VertexBufferLayout {
        location: 3,
        format: vk::Format::R32G32B32A32_SFLOAT,
        input_rate: vk::VertexInputRate::INSTANCE,
    },
];

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
        .expect("Failed to create window");

    let entry = unsafe { ash::Entry::load() }.unwrap();

    let app_info = vk::ApplicationInfo::builder()
        .application_name(c_str_macro::c_str!("KISS Graph"))
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_1);

    let instance_extensions = {
        let mut instance_extensions = ash_window::enumerate_required_extensions(&window)
            .unwrap()
            .to_vec();
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
        CStrList::new(vec![khr::Swapchain::name(), khr::DynamicRendering::name(), vk::ExtDescriptorIndexingFn::name()]);

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

        let mut descriptor_indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::builder()
            .runtime_descriptor_array(true)
            .descriptor_binding_partially_bound(true);

        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_info)
            .enabled_features(&device_features)
            .enabled_extension_names(required_extensions.pointers())
            .enabled_layer_names(enabled_layers.pointers())
            .push_next(&mut dynamic_rendering_features)
            .push_next(&mut descriptor_indexing_features);

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

    let mut device = kiss_engine_vk::Device::new(
        device,
        allocator,
        debug_utils_ext,
        swapchain_ext,
        queue_family,
    );

    let mut swapchain = Swapchain::new(&device, &swapchain_info);

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


    let size = window.inner_size();
    let mut perspective_matrix = perspective_matrix_reversed(size.width, size.height);

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

    let mut uniform_buffer = device.create_buffer(
        "uniform buffer",
        bytemuck::bytes_of(&Uniforms {
            matrices: perspective_matrix * view_matrix,
            player_position: Vec3::new(state.player_position.x, 0.0, state.player_position.y),
            player_facing: state.player_facing,
            camera_position: Vec3::new(state.player_position.x, 0.0, state.player_position.y)
                + player_height_offset
                + state.orbit.as_vector(),
            time: state.time,
            window_size: Vec2::new(extent.width as f32, extent.height as f32),
            ..Default::default()
        }),
        vk::BufferUsageFlags::UNIFORM_BUFFER,
    );

    let mut meteor_buffer = device.create_buffer(
        "meteor buffer",
        bytemuck::bytes_of(&MeteorGpuProps {
            position: state.meteor_props.position,
            _padding: 0,
        }),
        vk::BufferUsageFlags::UNIFORM_BUFFER,
    );

    let mut bounce_sphere_buffer = device.create_buffer(
        "bounce sphere buffer",
        bytemuck::bytes_of(&state.bounce_sphere_props),
        vk::BufferUsageFlags::UNIFORM_BUFFER,
    );
    let dynamic_rendering_ext = khr::DynamicRendering::new(&instance, &device);

    let height_map = device.get_image(
        "height map",
        vk::Extent2D {
            width: 1024,
            height: 1024,
        },
        vk::Format::R32G32B32A32_SFLOAT,
        vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC,
    );

    let buffer_size = 1024 * 1024 * 4 * 4;

    let target_buffer = device.create_buffer_of_size(
        "height map target buffer",
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST,
    );

    unsafe {
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

        let subres_layers = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);

        let extent = vk::Extent2D {
            width: 1024,
            height: 1024,
        };

        vk_sync::cmd::pipeline_barrier(
            &device,
            command_buffer,
            None,
            &[],
            &[vk_sync::ImageBarrier {
                previous_accesses: &[],
                next_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
                discard_contents: true,
                image: height_map.image,
                range: *subres,
                ..Default::default()
            }],
        );

        kiss_engine_vk::begin_rendering(
            &device,
            command_buffer,
            &dynamic_rendering_ext,
            &[kiss_engine_vk::Attachment {
                view: height_map.view,
                clear_value: kiss_engine_vk::ClearValue::Clear([0.0; 4]),
            }],
            None,
            extent,
        );

        let pipeline = device.get_graphics_pipeline(
            "bake pipeline",
            device.get_shader("shaders/compiled/bake_height_map.vert.spv"),
            Some(device.get_shader("shaders/compiled/bake_height_map.frag.spv")),
            &Default::default(),
            &[vk::Format::R32G32B32A32_SFLOAT],
            BASIC_FORMAT,
        );

        device.inner.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline,
        );

        plane.render(&device, command_buffer, 1);

        dynamic_rendering_ext.cmd_end_rendering(command_buffer);

        vk_sync::cmd::pipeline_barrier(
            &device,
            command_buffer,
            None,
            &[],
            &[vk_sync::ImageBarrier {
                previous_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
                next_accesses: &[vk_sync::AccessType::TransferRead],
                image: height_map.image,
                range: *subres,
                ..Default::default()
            }],
        );

        device.inner.cmd_copy_image_to_buffer(
            command_buffer,
            height_map.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            target_buffer.buffer,
            &[vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: *subres_layers,
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
            }],
        );

        vk_sync::cmd::pipeline_barrier(
            &device,
            command_buffer,
            None,
            &[],
            &[vk_sync::ImageBarrier {
                previous_accesses: &[vk_sync::AccessType::TransferRead],
                next_accesses: &[vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer],
                image: height_map.image,
                range: *subres,
                ..Default::default()
            }],
        );

        device.inner.end_command_buffer(command_buffer).unwrap();

        let fence_info = vk::FenceCreateInfo::builder();
        let fence = device.inner.create_fence(&fence_info, None).unwrap();

        device
            .inner
            .queue_submit(
                queue,
                &[*vk::SubmitInfo::builder()
                    .command_buffers(&[command_buffer])
                    .signal_semaphores(&[])],
                fence,
            )
            .unwrap();

        device
            .inner
            .wait_for_fences(&[fence], true, u64::MAX)
            .unwrap();
    }

    let heightmap = {
        let bytes = target_buffer.read_mapped();

        let vec4s: &[Vec4] = bytemuck::cast_slice(&bytes);

        HeightMap {
            floats: vec4s.iter().map(|vec4| vec4.x).collect(),
            height: 1024,
            width: 1024,
        }
    };

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

    let mut rng = rand::rngs::OsRng::default();
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

    let forest_points = device.create_buffer(
        "forest buffer",
        bytemuck::cast_slice(&forest_points),
        vk::BufferUsageFlags::VERTEX_BUFFER,
    );

    let house_points = device.create_buffer(
        "house buffer",
        bytemuck::cast_slice(&house_points),
        vk::BufferUsageFlags::VERTEX_BUFFER,
    );

    let grass_texture = load_image(
        &device,
        queue,
        include_bytes!("../grass.png"),
        "grass texture",
    );
    let sand_texture = load_image(
        &device,
        queue,
        include_bytes!("../sand.png"),
        "sand texture",
    );
    let rock_texture = load_image(
        &device,
        queue,
        include_bytes!("../rock.png"),
        "rock texture",
    );
    let forest_texture = load_image(
        &device,
        queue,
        include_bytes!("../grass_forest.png"),
        "forest texture",
    );
    let house_texture = load_image(
        &device,
        queue,
        include_bytes!("../house.png"),
        "house texture",
    );

    let forest_map_tex = load_image(
        &device,
        queue,
        include_bytes!("../forestmap.png"),
        "forest map texture",
    );

    let giant_tex = load_image(
        &device,
        queue,
        include_bytes!("../fire_giant.png"),
        "fire giant texture",
    );

    let meteor_tex = load_image(
        &device,
        queue,
        include_bytes!("../meteor.png"),
        "meteor texture",
    );

    let mut image_list = kiss_engine_vk::BindlessImageList::new(&device);

    image_list.push(grass_texture);
    image_list.push(sand_texture);
    image_list.push(rock_texture);
    image_list.push(forest_texture);
    image_list.push(forest_map_tex);
    let giant_tex = image_list.push(giant_tex);

    let sampler = device.create_resource(unsafe {
        device
            .inner
            .create_sampler(
                &vk::SamplerCreateInfo::builder().max_lod(vk::LOD_CLAMP_NONE),
                None,
            )
            .unwrap()
    });

    let linear_sampler = device.create_resource(unsafe {
        device
            .inner
            .create_sampler(
                &vk::SamplerCreateInfo::builder()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )
            .unwrap()
    });

    let mut joint_transforms_buffer = device.create_buffer_of_size(
        "player joint transforms",
        (capsule.joint_indices_to_node_indices.len()
            * std::mem::size_of::<gltf_helpers::Similarity>()) as u64,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
    );

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
            #[cfg(not(feature = "wasm"))]
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
                state.handle_key(key, key_state == event::ElementState::Pressed);
            }
            WindowEvent::Focused(false) => {
                state.kbd = Default::default();
            }
            _ => {}
        },
        event::Event::MainEventsCleared => {
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

            joint_transforms_buffer.write_mapped(bytemuck::cast_slice(&joint_transforms), 0);

            meteor_buffer.write_mapped(
                bytemuck::bytes_of(&MeteorGpuProps {
                    position: state.meteor_props.position,
                    _padding: 0,
                }),
                0,
            );

            bounce_sphere_buffer.write_mapped(bytemuck::bytes_of(&state.bounce_sphere_props), 0);

            let player_position_3d = state.player_position_3d(&heightmap);

            uniform_buffer.write_mapped(
                bytemuck::bytes_of(&Uniforms {
                    matrices: {
                        let view_matrix = {
                            Mat4::look_at_rh(
                                player_position_3d + player_height_offset + state.orbit.as_vector(),
                                player_position_3d + player_height_offset,
                                Vec3::Y,
                            )
                        };

                        perspective_matrix * view_matrix
                    },
                    player_position: player_position_3d,
                    player_facing: state.player_facing,
                    camera_position: player_position_3d
                        + player_height_offset
                        + state.orbit.as_vector(),
                    time: state.time,
                    window_size: Vec2::new(extent.width as f32, extent.height as f32),
                    ..Default::default()
                }),
                0,
            );
            window.request_redraw();
        }
        event::Event::RedrawRequested(_) => {
            unsafe {
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

                let opaque_texture = device.get_image(
                    "opaque texture",
                    extent,
                    vk::Format::R8G8B8A8_SRGB,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED
                );

                let depth_subres = vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .level_count(1)
                    .layer_count(1);

                vk_sync::cmd::pipeline_barrier(
                    &device,
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
                        vk_sync::ImageBarrier {
                            previous_accesses: &[vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer],
                            next_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
                            discard_contents: true,
                            image: opaque_texture.image,
                            range: *subres,
                            ..Default::default()
                        },
                    ],
                );

                kiss_engine_vk::begin_rendering(
                    &device,
                    command_buffer,
                    &dynamic_rendering_ext,
                    &[kiss_engine_vk::Attachment {
                        view: swapchain.views[swapchain_image_index as usize],
                        clear_value: kiss_engine_vk::ClearValue::Clear([0.1, 0.2, 0.3, 1.0]),
                    }, kiss_engine_vk::Attachment {
                        view: opaque_texture.view,
                        clear_value: kiss_engine_vk::ClearValue::Clear([0.1, 0.2, 0.3, 1.0]),
                    }],
                    Some(kiss_engine_vk::Attachment {
                        view: depth_buffer.view,
                        clear_value: kiss_engine_vk::ClearValue::Clear(0.0),
                    }),
                    extent,
                );

                {
                    let formats = &[swapchain_info.image_format, vk::Format::R8G8B8A8_SRGB];
                    let device = device.rendering_with(formats);

                    let pipeline = device.get_graphics_pipeline(
                        "plane pipeline",
                        device.get_shader("shaders/compiled/plane.vert.spv"),
                        Some(device.get_shader("shaders/compiled/plane.frag.spv")),
                        &Default::default(),
                        BASIC_FORMAT,
                    );

                    let ds = device.get_descriptor_set(
                        "plane",
                        pipeline,
                        &[
                            BindingResource::Buffer(&uniform_buffer),
                            BindingResource::Buffer(&meteor_buffer),
                            BindingResource::Sampler(&sampler),
                            BindingResource::Sampler(&linear_sampler),
                            BindingResource::BindlessImageList(&image_list)
                        ],
                    );

                    device.inner.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.pipeline,
                    );
                    device.inner.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.layout,
                        0,
                        &[ds.inner],
                        &[],
                    );

                    plane.render(&device, command_buffer, 1);

                    {
                        let pipeline = device.get_graphics_pipeline(
                            "player pipeline",
                            device.get_shader("shaders/compiled/capsule.vert.spv"),
                            Some(device.get_shader("shaders/compiled/tree.frag.spv")),
                            &Default::default(),
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
                                VertexBufferLayout {
                                    location: 2,
                                    format: vk::Format::R32G32_SFLOAT,
                                    input_rate: vk::VertexInputRate::VERTEX,
                                },
                                VertexBufferLayout {
                                    location: 3,
                                    format: vk::Format::R16G16B16A16_UINT,
                                    input_rate: vk::VertexInputRate::VERTEX,
                                },
                                VertexBufferLayout {
                                    location: 4,
                                    format: vk::Format::R32G32B32A32_SFLOAT,
                                    input_rate: vk::VertexInputRate::VERTEX,
                                },
                            ],
                        );

                        let ds = device.get_descriptor_set(
                            "capsule",
                            pipeline,
                            &[
                                BindingResource::Buffer(&uniform_buffer),
                                BindingResource::Buffer(&meteor_buffer),
                                BindingResource::Sampler(&sampler),
                                BindingResource::Texture(image_list.get(giant_tex)),
                                BindingResource::Buffer(&joint_transforms_buffer),
                            ],
                        );

                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.pipeline,
                        );
                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.layout,
                            0,
                            &[ds.inner],
                            &[],
                        );

                        capsule.render_animated(&device, command_buffer, 1);
                    }

                    dynamic_rendering_ext.cmd_end_rendering(command_buffer);
                }

                kiss_engine_vk::begin_rendering(
                    &device,
                    command_buffer,
                    &dynamic_rendering_ext,
                    &[kiss_engine_vk::Attachment {
                        view: swapchain.views[swapchain_image_index as usize],
                        clear_value: kiss_engine_vk::ClearValue::Load,
                    }],
                    Some(kiss_engine_vk::Attachment {
                        view: depth_buffer.view,
                        clear_value: kiss_engine_vk::ClearValue::Load,
                    }),
                    extent,
                );

                vk_sync::cmd::pipeline_barrier(
                    &device,
                    command_buffer,
                    None,
                    &[],
                    &[vk_sync::ImageBarrier {
                        previous_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
                        next_accesses: &[vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer],
                        image: opaque_texture.image,
                        range: *subres,
                        ..Default::default()
                    }],
                );

                {
                    let formats = &[swapchain_info.image_format];
                    let device = device.rendering_with(formats);


                let pipeline = device.get_graphics_pipeline(
                    "water pipeline",
                    device.get_shader(
                        "shaders/compiled/plane.vert.spv",
                    ),
                    Some(device.get_shader(
                        "shaders/compiled/water.frag.spv",
                    )),
                    &Default::default(),
                    BASIC_FORMAT,
                );

                let height_map = device.try_get_cached_image("height map").unwrap();

                let descriptor_set = device.get_descriptor_set(
                    "water",
                    pipeline,
                    &[
                        BindingResource::Buffer(&uniform_buffer),
                        BindingResource::Texture(opaque_texture),
                        BindingResource::Sampler(&sampler),
                        BindingResource::Texture(&height_map),
                        BindingResource::Buffer(&meteor_buffer),
                    ],
                );

                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.layout,
                    0,
                    &[descriptor_set.inner],
                    &[],
                );

                water.render(&device, command_buffer, 1);

                /*
                render_pass.set_pipeline(&pipeline.pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);

                water.render(&mut render_pass, 1);
                */
                }

                dynamic_rendering_ext.cmd_end_rendering(command_buffer);


                vk_sync::cmd::pipeline_barrier(
                    &device,
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

                device.end_command_buffer(command_buffer).unwrap();

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
            }

            /*let frame = match surface.get_current_texture() {
                Ok(frame) => frame,
                Err(_) => {
                    surface.configure(&device, &config);
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
                            format: vk::Format::R32G32B32_SFLOAT,
                            input_rate: vk::VertexInputRate::VERTEX,
                        },
                        VertexBufferLayout {
                            location: 1,
                            format: vk::Format::R32G32B32_SFLOAT,
                            input_rate: vk::VertexInputRate::VERTEX,
                        },
                        VertexBufferLayout {
                            location: 2,
                            format: vk::Format::R32G32_SFLOAT,
                            input_rate: vk::VertexInputRate::VERTEX,
                        },
                        VertexBufferLayout {
                            location: 3,
                            format: wgpu::VertexFormat::Uint16x4,
                            input_rate: vk::VertexInputRate::VERTEX,
                        },
                        VertexBufferLayout {
                            location: 4,
                            format: vk::Format::R32G32B32A32_SFLOAT,
                            input_rate: vk::VertexInputRate::VERTEX,
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
                        &device,
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

            device.flush();*/
        }
        _ => {}
    });

    unsafe {
        device.queue_wait_idle(queue).unwrap();

        device.destroy_semaphore(present_semaphore, None);
        device.destroy_semaphore(render_semaphore, None);
        device.destroy_fence(render_fence, None);
        device.destroy_command_pool(command_pool, None);
    }

    drop(uniform_buffer);
    drop(swapchain);

    let raw_device = device.inner.clone();

    /*drop(device);

    unsafe {
        raw_device.destroy_device(None);

        surface_ext.destroy_surface(surface, None);

        instance.destroy_instance(None);
    }*/
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

fn tick(supported: bool) -> &'static str {
    if supported {
        "✔️"
    } else {
        "❌"
    }
}
