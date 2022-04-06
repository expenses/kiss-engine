pub mod binding_resources;
pub mod device;
pub mod pipeline_resources;
pub mod primitives;

use ash::vk;

pub use device::Device;
pub use pipeline_resources::VertexBufferLayout;

use std::ffi::CStr;
use std::os::raw::c_char;

pub unsafe fn cstr_from_array(array: &[c_char]) -> &CStr {
    CStr::from_ptr(array.as_ptr())
}

pub use binding_resources::{BindlessImageList, Buffer, Image, Resource};

pub struct Attachment<T> {
    pub view: vk::ImageView,
    pub clear_value: ClearValue<T>,
}

pub enum ClearValue<T> {
    Load,
    Clear(T),
}

pub unsafe fn begin_rendering(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    dynamic_rendering: &ash::extensions::khr::DynamicRendering,
    color_attachments: &[Attachment<[f32; 4]>],
    depth_attachment: Option<Attachment<f32>>,
    extent: vk::Extent2D,
) {
    let color_attachments: Vec<_> = color_attachments
        .iter()
        .map(|attachment| {
            let (load_op, clear_value) = match attachment.clear_value {
                ClearValue::Load => (vk::AttachmentLoadOp::LOAD, vk::ClearValue::default()),
                ClearValue::Clear(clear) => (
                    vk::AttachmentLoadOp::CLEAR,
                    vk::ClearValue {
                        color: vk::ClearColorValue { float32: clear },
                    },
                ),
            };

            vk::RenderingAttachmentInfo {
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                image_view: attachment.view,
                load_op,
                clear_value,
                ..Default::default()
            }
        })
        .collect();

    let depth_attachment = depth_attachment.map(|attachment| {
        let (load_op, clear_value) = match attachment.clear_value {
            ClearValue::Load => (vk::AttachmentLoadOp::LOAD, vk::ClearValue::default()),
            ClearValue::Clear(clear) => (
                vk::AttachmentLoadOp::CLEAR,
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: clear,
                        stencil: 0,
                    },
                },
            ),
        };

        vk::RenderingAttachmentInfo {
            image_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            image_view: attachment.view,
            load_op,
            clear_value,
            ..Default::default()
        }
    });

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

    device.cmd_set_scissor(command_buffer, 0, &[area]);
    device.cmd_set_viewport(command_buffer, 0, &[viewport]);

    let mut rendering_info = vk::RenderingInfoKHR::builder()
        .layer_count(1)
        .render_area(area)
        .color_attachments(&color_attachments);

    if let Some(depth_attachment) = depth_attachment.as_ref() {
        rendering_info = rendering_info.depth_attachment(depth_attachment);
    }

    dynamic_rendering.cmd_begin_rendering(command_buffer, &rendering_info);
}
