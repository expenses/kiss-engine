use crate::device::{set_object_name, Device};
use ash::extensions::ext;
use ash::vk;
use std::sync::atomic;

pub const MAX_BINDLESS_IMAGES: u32 = u16::max_value() as u32;

pub(crate) struct Shader {
    device: ash::Device,
    pub(crate) module: vk::ShaderModule,
    descriptor_set_layout_bindings: Vec<DescriptorSetLayoutBinding>,
}

impl Shader {
    pub(crate) fn load(
        device: &ash::Device,
        debug_utils: &ext::DebugUtils,
        filename: &str,
    ) -> Self {
        let bytes = std::fs::read(filename).unwrap();

        let reflection = rspirv_reflect::Reflection::new_from_spirv(&bytes).unwrap();
        let descriptor_set_layout_bindings = reflect_descriptor_set_layout_bindings(&reflection);

        let spirv = ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap();

        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&spirv), None)
        }
        .unwrap();

        set_object_name(device, debug_utils, module, filename);

        Self {
            device: device.clone(),
            module,
            descriptor_set_layout_bindings,
        }
    }

    fn merge_bindings(&self, other: &Shader) -> Vec<DescriptorSetLayoutBinding> {
        merge_descriptor_set_layout_bindings(
            &self.descriptor_set_layout_bindings,
            &other.descriptor_set_layout_bindings,
        )
    }

    fn solo_bindings(&self) -> Vec<DescriptorSetLayoutBinding> {
        self.descriptor_set_layout_bindings.clone()
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.module, None);
        }
    }
}

// Minus p_immutable_samplers as it's not Send and has a partially bound flag.
#[derive(Clone, Copy, Debug)]
struct DescriptorSetLayoutBinding {
    binding: u32,
    stage_flags: vk::ShaderStageFlags,
    descriptor_count: u32,
    descriptor_type: vk::DescriptorType,
    partially_bound: bool,
}

impl DescriptorSetLayoutBinding {
    fn as_vk(&self) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding: self.binding,
            stage_flags: self.stage_flags,
            descriptor_count: self.descriptor_count,
            descriptor_type: self.descriptor_type,
            p_immutable_samplers: std::ptr::null(),
        }
    }
}

fn merge_descriptor_set_layout_bindings(
    a: &[DescriptorSetLayoutBinding],
    b: &[DescriptorSetLayoutBinding],
) -> Vec<DescriptorSetLayoutBinding> {
    let mut merged = a.to_vec();

    for merging_entry in b {
        if let Some(entry) = merged
            .iter_mut()
            .find(|entry| entry.binding == merging_entry.binding)
        {
            entry.stage_flags |= merging_entry.stage_flags;
        } else {
            merged.push(*merging_entry);
        }
    }

    merged
}

pub struct ReloadableShader {
    pub(crate) shader: parking_lot::Mutex<Shader>,
    pub(crate) version: std::sync::atomic::AtomicU32,
}

impl ReloadableShader {
    pub(crate) fn get_version(&self) -> u32 {
        self.version.load(atomic::Ordering::Relaxed)
    }
}

fn reflect_shader_stage_flags(reflection: &rspirv_reflect::Reflection) -> vk::ShaderStageFlags {
    let entry_point_inst = &reflection.0.entry_points[0];

    let execution_model = entry_point_inst.operands[0].unwrap_execution_model();

    match execution_model {
        rspirv_reflect::spirv::ExecutionModel::Vertex => vk::ShaderStageFlags::VERTEX,
        rspirv_reflect::spirv::ExecutionModel::Fragment => vk::ShaderStageFlags::FRAGMENT,
        rspirv_reflect::spirv::ExecutionModel::GLCompute => vk::ShaderStageFlags::COMPUTE,
        other => unimplemented!("{:?}", other),
    }
}

fn reflect_descriptor_set_layout_bindings(
    reflection: &rspirv_reflect::Reflection,
) -> Vec<DescriptorSetLayoutBinding> {
    let shader_stage_flags = reflect_shader_stage_flags(reflection);

    let descriptor_sets = reflection.get_descriptor_sets().unwrap();

    if descriptor_sets.is_empty() {
        return Vec::new();
    }

    if descriptor_sets.len() > 1 {
        panic!(
            "Expected <= 1 descriptor set; got {}",
            descriptor_sets.len()
        );
    }

    let descriptor_set = &descriptor_sets[&0];

    descriptor_set
        .iter()
        .map(|(&binding, info)| {
            let (descriptor_count, partially_bound) = match info.binding_count {
                rspirv_reflect::BindingCount::One => (1, false),
                rspirv_reflect::BindingCount::StaticSized(size) => (size as u32, false),
                rspirv_reflect::BindingCount::Unbounded => (MAX_BINDLESS_IMAGES, true),
            };

            DescriptorSetLayoutBinding {
                binding,
                stage_flags: shader_stage_flags,
                descriptor_count,
                descriptor_type: vk::DescriptorType::from_raw(info.ty.0 as i32),
                partially_bound,
            }
        })
        .collect()
}

pub struct GraphicsPipelineSettings {
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_compare_op: vk::CompareOp,
}

impl Default for GraphicsPipelineSettings {
    fn default() -> Self {
        Self {
            depth_test_enable: true,
            depth_write_enable: true,
            depth_compare_op: vk::CompareOp::GREATER,
        }
    }
}

pub struct GraphicsPipeline {
    device: ash::Device,
    pub(crate) descriptor_set_layout: vk::DescriptorSetLayout,
    pub layout: vk::PipelineLayout,
    pub(crate) shader_versions: (u32, Option<u32>),
    pub pipeline: vk::Pipeline,
}

impl GraphicsPipeline {
    pub(crate) fn new<F: std::ops::Deref<Target = Shader>>(
        device: &Device,
        name: &str,
        vertex_shader: &Shader,
        fragment_shader: Option<F>,
        settings: &GraphicsPipelineSettings,
        color_attachment_formats: &[vk::Format],
        vertex_binding_layouts: &[VertexBufferLayout],
        shader_versions: (u32, Option<u32>),
    ) -> Self {
        let bindings = if let Some(fragment_shader) = fragment_shader.as_ref() {
            vertex_shader.merge_bindings(fragment_shader)
        } else {
            vertex_shader.solo_bindings()
        };

        let vk_bindings: Vec<_> = bindings.iter().map(|binding| binding.as_vk()).collect();

        let any_partially_bound = bindings.iter().any(|binding| binding.partially_bound);

        let flags: Vec<_> = bindings
            .iter()
            .map(|binding| {
                if binding.partially_bound {
                    vk::DescriptorBindingFlags::PARTIALLY_BOUND
                } else {
                    vk::DescriptorBindingFlags::empty()
                }
            })
            .collect();

        let mut flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().binding_flags(&flags);

        let mut create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&vk_bindings);

        if any_partially_bound {
            create_info = create_info.push_next(&mut flags);
        }

        let descriptor_set_layout = unsafe {
            device
                .inner
                .create_descriptor_set_layout(&create_info, None)
        }
        .unwrap();

        device.set_object_name(
            descriptor_set_layout,
            &format!("{} descriptor set layout", name),
        );

        let pipeline_layout = unsafe {
            device.inner.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder().set_layouts(&[descriptor_set_layout]),
                None,
            )
        }
        .unwrap();

        let stages = if let Some(fragment_shader) = fragment_shader {
            vec![
                *vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .name(c_str_macro::c_str!("main"))
                    .module(vertex_shader.module),
                *vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .name(c_str_macro::c_str!("main"))
                    .module(fragment_shader.module),
            ]
        } else {
            vec![*vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .name(c_str_macro::c_str!("main"))
                .module(vertex_shader.module)]
        };

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let vertex_binding_descriptions: Vec<_> = vertex_binding_layouts
            .iter()
            .map(|layout| vk::VertexInputBindingDescription {
                binding: layout.location,
                stride: match layout.format {
                    vk::Format::R32G32_SFLOAT => 8,
                    vk::Format::R32G32B32_SFLOAT => 12,
                    vk::Format::R32G32B32A32_SFLOAT => 16,
                    vk::Format::R16G16B16A16_UINT => 8,
                    other => unimplemented!("format: {:?}", other),
                },
                input_rate: layout.input_rate,
            })
            .collect();

        let vertex_attribute_descriptions: Vec<_> = vertex_binding_layouts
            .iter()
            .map(|layout| vk::VertexInputAttributeDescription {
                location: layout.location,
                format: layout.format,
                binding: layout.location,
                offset: 0,
            })
            .collect();

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vertex_binding_descriptions)
            .vertex_attribute_descriptions(&vertex_attribute_descriptions);

        let rasterisation_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        let blend_states: Vec<_> = color_attachment_formats
            .iter()
            .map(|_| {
                *vk::PipelineColorBlendAttachmentState::builder()
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                    .blend_enable(false)
            })
            .collect();

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(&blend_states);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(settings.depth_test_enable)
            .depth_write_enable(settings.depth_write_enable)
            .depth_compare_op(settings.depth_compare_op);

        let mut pipeline_rendering = vk::PipelineRenderingCreateInfoKHR::builder()
            .color_attachment_formats(color_attachment_formats)
            .depth_attachment_format(vk::Format::D32_SFLOAT);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .layout(pipeline_layout)
            .stages(&stages)
            .input_assembly_state(&input_assembly_state)
            .vertex_input_state(&vertex_input_state)
            .rasterization_state(&rasterisation_state)
            .multisample_state(&multisample_state)
            .viewport_state(&viewport_state)
            .dynamic_state(&dynamic_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .push_next(&mut pipeline_rendering);

        let graphics_pipeline = unsafe {
            device.inner.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[*pipeline_info],
                None,
            )
        }
        .unwrap()[0];

        Self {
            device: device.inner.clone(),
            descriptor_set_layout,
            layout: pipeline_layout,
            shader_versions,
            pipeline: graphics_pipeline,
        }
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

pub struct VertexBufferLayout {
    pub location: u32,
    pub format: vk::Format,
    pub input_rate: vk::VertexInputRate,
}
