use std::sync::atomic;
use std::sync::Arc;

pub struct RenderPipeline {
    pub pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    shader_versions: (u32, Option<u32>),
}

fn reflect_shader_stages(reflection: &rspirv_reflect::Reflection) -> wgpu::ShaderStages {
    let entry_point_inst = &reflection.0.entry_points[0];

    let execution_model = entry_point_inst.operands[0].unwrap_execution_model();

    match execution_model {
        rspirv_reflect::rspirv::spirv::ExecutionModel::Vertex => wgpu::ShaderStages::VERTEX,
        rspirv_reflect::rspirv::spirv::ExecutionModel::Fragment => wgpu::ShaderStages::FRAGMENT,
        rspirv_reflect::rspirv::spirv::ExecutionModel::GLCompute => wgpu::ShaderStages::COMPUTE,
        other => unimplemented!("{:?}", other),
    }
}

fn reflect_bind_group_layout_entries(
    reflection: &rspirv_reflect::Reflection,
) -> Vec<wgpu::BindGroupLayoutEntry> {
    let shader_stages = reflect_shader_stages(reflection);

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
        .map(|(&binding, info)| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: shader_stages,
            count: match info.binding_count {
                rspirv_reflect::BindingCount::One => None,
                rspirv_reflect::BindingCount::StaticSized(size) => {
                    Some(std::num::NonZeroU32::new(size as u32).unwrap())
                }
                rspirv_reflect::BindingCount::Unbounded => {
                    unimplemented!("No good way to handle unbounded binding counts yet.")
                }
            },
            ty: match info.ty {
                rspirv_reflect::DescriptorType::UNIFORM_BUFFER => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                rspirv_reflect::DescriptorType::SAMPLER => {
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
                }
                rspirv_reflect::DescriptorType::SAMPLED_IMAGE => wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                rspirv_reflect::DescriptorType::STORAGE_BUFFER => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                _ => unimplemented!("{:?}", info.ty),
            },
        })
        .collect()
}

fn merge_bind_group_layout_entries(
    a: &[wgpu::BindGroupLayoutEntry],
    b: &[wgpu::BindGroupLayoutEntry],
) -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut merged = a.to_vec();

    for merging_entry in b {
        if let Some(entry) = merged
            .iter_mut()
            .find(|entry| entry.binding == merging_entry.binding)
        {
            entry.visibility |= merging_entry.visibility;
        } else {
            merged.push(*merging_entry);
        }
    }

    merged
}

pub struct VertexBufferLayout {
    pub location: u32,
    pub format: wgpu::VertexFormat,
    pub step_mode: wgpu::VertexStepMode,
}

impl VertexBufferLayout {
    fn attribute_array(&self) -> [wgpu::VertexAttribute; 1] {
        [wgpu::VertexAttribute {
            format: self.format,
            offset: 0,
            shader_location: self.location,
        }]
    }
}

#[derive(Clone)]
pub struct RenderPipelineDesc {
    pub primitive: wgpu::PrimitiveState,
    pub depth_write_enabled: bool,
    pub depth_compare: wgpu::CompareFunction,
    pub multisample: wgpu::MultisampleState,
    pub multiview: Option<std::num::NonZeroU32>,
    pub blend: Option<wgpu::BlendState>,
}

impl Default for RenderPipelineDesc {
    fn default() -> Self {
        Self {
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Greater,
            multisample: Default::default(),
            multiview: None,
            blend: None,
        }
    }
}

fn create_render_pipeline(
    device: &wgpu::Device,
    name: &str,
    vertex_shader: &Shader,
    fragment_shader: &Shader,
    render_pipeline_desc: RenderPipelineDesc,
    vertex_buffer_layout: &[VertexBufferLayout],
    attachment_texture_format: wgpu::TextureFormat,
    depth_stencil_attachment_format: Option<wgpu::TextureFormat>,
    shader_versions: (u32, Option<u32>),
) -> RenderPipeline {
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{} bind group layout", name)),
        entries: &merge_bind_group_layout_entries(
            &vertex_shader.reflected_bind_group_layout_entries,
            &fragment_shader.reflected_bind_group_layout_entries,
        ),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{} layout", name)),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let vertex_attribute_arrays = vertex_buffer_layout
        .iter()
        .map(|layout| layout.attribute_array())
        .collect::<Vec<_>>();

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(name),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vertex_shader.module,
            entry_point: "main",
            buffers: &{
                vertex_buffer_layout
                    .iter()
                    .enumerate()
                    .map(|(i, layout)| wgpu::VertexBufferLayout {
                        array_stride: layout.format.size(),
                        step_mode: layout.step_mode,
                        attributes: &vertex_attribute_arrays[i],
                    })
                    .collect::<Vec<_>>()
            },
        },
        fragment: Some(wgpu::FragmentState {
            module: &fragment_shader.module,
            entry_point: "main",
            targets: &[wgpu::ColorTargetState {
                format: attachment_texture_format,
                blend: render_pipeline_desc.blend,
                write_mask: wgpu::ColorWrites::ALL,
            }],
        }),
        primitive: render_pipeline_desc.primitive,
        depth_stencil: depth_stencil_attachment_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: render_pipeline_desc.depth_write_enabled,
            depth_compare: render_pipeline_desc.depth_compare,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: render_pipeline_desc.multisample,
        multiview: render_pipeline_desc.multiview,
    });

    RenderPipeline {
        pipeline: render_pipeline,
        bind_group_layout,
        shader_versions,
    }
}

struct CacheMap<K, V> {
    map: std::collections::HashMap<K, Box<V>>,
    inserting: elsa::FrozenMap<K, Box<V>>,
}

impl<K: Eq + std::hash::Hash, V> Default for CacheMap<K, V> {
    fn default() -> Self {
        Self {
            map: Default::default(),
            inserting: Default::default(),
        }
    }
}

impl<K, V> CacheMap<K, V>
where
    K: Eq + std::hash::Hash,
{
    fn try_get(&self, key: &K) -> Option<&V> {
        if let Some(value) = self.inserting.get(key) {
            Some(value)
        } else if let Some(value) = self.map.get(key) {
            Some(value)
        } else {
            None
        }
    }

    fn get_or_insert_with<F: FnOnce() -> V>(&self, key: K, func: F) -> &V {
        // Check inserting first, in case we're updating something.
        if let Some(value) = self.inserting.get(&key) {
            value
        } else if let Some(value) = self.map.get(&key) {
            value
        } else {
            self.inserting.insert(key, Box::new(func()))
        }
    }

    fn insert_or_replace(&self, key: K, value: V) -> &V {
        self.inserting.insert(key, Box::new(value))
    }

    fn flush(&mut self, name: &str) {
        let inserting = std::mem::take(&mut self.inserting);

        let old_len = self.map.len();
        let mut num_updating = 0;

        for (key, value) in inserting.into_map().into_iter() {
            self.map.insert(key, value);
            num_updating += 1;
        }

        let new_len = self.map.len();

        if new_len != old_len {
            log::info!(
                "Adding {} item(s) to {} ({} -> {})",
                new_len - old_len,
                name,
                old_len,
                new_len
            );
        } else if num_updating > 0 {
            log::info!("Updating {} items in {} ({})", num_updating, name, new_len);
        }
    }
}

struct Shader {
    module: wgpu::ShaderModule,
    reflected_bind_group_layout_entries: Vec<wgpu::BindGroupLayoutEntry>,
}

impl Shader {
    fn load(device: &wgpu::Device, filename: &str, bytes: &[u8]) -> Self {
        Self {
            module: device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: Some(filename),
                source: wgpu::util::make_spirv(bytes),
            }),
            reflected_bind_group_layout_entries: {
                let reflection = rspirv_reflect::Reflection::new_from_spirv(bytes).unwrap();
                reflect_bind_group_layout_entries(&reflection)
            },
        }
    }
}

pub struct ReloadableShader {
    shader: parking_lot::Mutex<Shader>,
    version: std::sync::atomic::AtomicU32,
}

impl ReloadableShader {
    fn get_version(&self) -> u32 {
        self.version.load(atomic::Ordering::Relaxed)
    }
}

struct BindGroup {
    inner: wgpu::BindGroup,
    ids: Vec<u32>,
}

pub struct Resource<T> {
    inner: T,
    id: u32,
}

impl<T> std::ops::Deref for Resource<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub enum BindingResource<'a> {
    Buffer(&'a Resource<wgpu::Buffer>),
    Sampler(&'a Resource<wgpu::Sampler>),
    Texture(&'a Resource<TextureInner>),
}

impl<'a> BindingResource<'a> {
    fn id(&self) -> u32 {
        match self {
            Self::Buffer(res) => res.id,
            Self::Sampler(res) => res.id,
            Self::Texture(res) => res.id,
        }
    }
}

pub struct TextureInner {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

struct Texture {
    texture: Resource<TextureInner>,
    extent: wgpu::Extent3d,
}

pub struct Device {
    pub inner: Arc<wgpu::Device>,
    pipelines: CacheMap<&'static str, RenderPipeline>,
    bind_groups: CacheMap<&'static str, BindGroup>,
    shaders: CacheMap<&'static str, Arc<ReloadableShader>>,
    textures: CacheMap<&'static str, Texture>,
    pub next_resource_id: std::sync::atomic::AtomicU32,
}

impl Device {
    pub fn new(device: wgpu::Device) -> Self {
        Self {
            inner: Arc::new(device),
            pipelines: Default::default(),
            bind_groups: Default::default(),
            shaders: Default::default(),
            textures: Default::default(),
            next_resource_id: Default::default(),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_shader(&self, filename: &'static str) -> &ReloadableShader {
        self.shaders.get_or_insert_with(filename, || {
            let bytes = std::fs::read(filename).unwrap();

            let shader = Arc::new(ReloadableShader {
                shader: parking_lot::Mutex::new(Shader::load(&self.inner, filename, &bytes)),
                version: Default::default(),
            });

            std::thread::spawn({
                let shader = shader.clone();
                let device = self.inner.clone();
                move || {
                    use notify::Watcher;

                    let (tx, rx) = std::sync::mpsc::channel();

                    let mut watcher = notify::raw_watcher(tx).unwrap();

                    watcher
                        .watch(filename, notify::RecursiveMode::NonRecursive)
                        .unwrap();

                    for _ in rx.iter() {
                        log::info!("Reloading {}", filename);
                        let bytes = std::fs::read(filename).unwrap();
                        *shader.shader.lock() = Shader::load(&device, filename, &bytes);
                        shader
                            .version
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }

                    shader
                }
            });

            shader
        })
    }

    #[cfg(target_arch = "wasm32")]
    pub fn get_shader(&self, filename: &'static str, bytes: &[u8]) -> &ReloadableShader {
        self.shaders.get_or_insert_with(filename, || {
            let shader = Arc::new(ReloadableShader {
                shader: parking_lot::Mutex::new(Shader::load(&self.inner, filename, &bytes)),
                version: Default::default(),
            });

            shader
        })
    }

    fn get_pipeline(
        &self,
        name: &'static str,
        vertex_shader: &ReloadableShader,
        fragment_shader: &ReloadableShader,
        render_pipeline_desc: RenderPipelineDesc,
        buffer_layout: &[VertexBufferLayout],
        attachment_texture_format: wgpu::TextureFormat,
        depth_stencil_attachment_format: Option<wgpu::TextureFormat>,
    ) -> &RenderPipeline {
        let shader_versions = (
            vertex_shader.get_version(),
            Some(fragment_shader.get_version()),
        );

        let pipeline_fn = || {
            create_render_pipeline(
                &self.inner,
                name,
                &vertex_shader.shader.lock(),
                &fragment_shader.shader.lock(),
                render_pipeline_desc,
                buffer_layout,
                attachment_texture_format,
                depth_stencil_attachment_format,
                shader_versions,
            )
        };

        let pipeline = self.pipelines.get_or_insert_with(name, pipeline_fn.clone());

        if pipeline.shader_versions != shader_versions {
            self.pipelines.insert_or_replace(name, pipeline_fn())
        } else {
            pipeline
        }
    }

    fn get_bind_group<F: Fn() -> wgpu::BindGroup>(
        &self,
        name: &'static str,
        func: F,
        ids: &[u32],
    ) -> &wgpu::BindGroup {
        let bind_group = self.bind_groups.get_or_insert_with(name, || BindGroup {
            inner: func(),
            ids: ids.to_vec(),
        });

        if bind_group.ids != ids {
            &self
                .bind_groups
                .insert_or_replace(
                    name,
                    BindGroup {
                        inner: func(),
                        ids: ids.to_vec(),
                    },
                )
                .inner
        } else {
            &bind_group.inner
        }
    }

    pub fn create_resource<T>(&self, resource: T) -> Resource<T> {
        Resource {
            inner: resource,
            id: self
                .next_resource_id
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }

    pub fn flush(&mut self) {
        self.bind_groups.flush("bind groups");
        self.pipelines.flush("pipelines");
        self.shaders.flush("shaders");
        self.textures.flush("textures");
    }

    pub fn with_formats(
        &self,
        attachment_texture_format: wgpu::TextureFormat,
        depth_stencil_attachment_format: Option<wgpu::TextureFormat>,
    ) -> DeviceWithFormats {
        DeviceWithFormats {
            device: self,
            attachment_texture_format,
            depth_stencil_attachment_format,
        }
    }

    pub fn get_texture(
        &self,
        descriptor: &wgpu::TextureDescriptor<'static>,
    ) -> &Resource<TextureInner> {
        let name = descriptor.label.unwrap();

        let create_texture_fn = || Texture {
            texture: self.create_resource({
                let texture = self.inner.create_texture(descriptor);

                TextureInner {
                    view: texture.create_view(&Default::default()),
                    texture,
                }
            }),
            extent: descriptor.size,
        };

        let texture = self.textures.get_or_insert_with(name, create_texture_fn);

        if descriptor.size != texture.extent {
            let texture = self.textures.insert_or_replace(name, create_texture_fn());

            &texture.texture
        } else {
            &texture.texture
        }
    }

    pub fn try_get_cached_texture(&self, name: &'static str) -> Option<&Resource<TextureInner>> {
        self.textures.try_get(&name).map(|tex| &tex.texture)
    }
}

pub struct DeviceWithFormats<'a> {
    pub device: &'a Device,
    attachment_texture_format: wgpu::TextureFormat,
    depth_stencil_attachment_format: Option<wgpu::TextureFormat>,
}

impl<'a> DeviceWithFormats<'a> {
    pub fn get_pipeline(
        &self,
        name: &'static str,
        vertex_shader: &ReloadableShader,
        fragment_shader: &ReloadableShader,
        render_pipeline_desc: RenderPipelineDesc,
        buffer_layout: &[VertexBufferLayout],
    ) -> &'a RenderPipeline {
        self.device.get_pipeline(
            name,
            vertex_shader,
            fragment_shader,
            render_pipeline_desc,
            buffer_layout,
            self.attachment_texture_format,
            self.depth_stencil_attachment_format,
        )
    }

    pub fn get_bind_group(
        &self,
        name: &'static str,
        pipeline: &RenderPipeline,
        binding_resources: &[BindingResource],
    ) -> &'a wgpu::BindGroup {
        let ids: Vec<_> = binding_resources
            .iter()
            .map(|resource| resource.id())
            .collect();

        self.device.get_bind_group(
            name,
            || {
                self.device
                    .inner
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!("{} bind group", name)),
                        layout: &pipeline.bind_group_layout,
                        entries: &{
                            binding_resources
                                .iter()
                                .enumerate()
                                .map(|(binding, res)| wgpu::BindGroupEntry {
                                    binding: binding as u32,
                                    resource: match res {
                                        BindingResource::Sampler(res) => {
                                            wgpu::BindingResource::Sampler(&res.inner)
                                        }
                                        BindingResource::Texture(res) => {
                                            wgpu::BindingResource::TextureView(&res.inner.view)
                                        }
                                        BindingResource::Buffer(res) => {
                                            res.inner.as_entire_binding()
                                        }
                                    },
                                })
                                .collect::<Vec<_>>()
                        },
                    })
            },
            &ids,
        )
    }
}
