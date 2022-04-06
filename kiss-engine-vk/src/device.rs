use ash::extensions::{ext, khr};
use ash::vk;
use futures::{SinkExt, StreamExt};
use std::ffi::CString;
use std::sync::Arc;

use crate::binding_resources::{DescriptorSet, Image, Resource};
use crate::pipeline_resources::{
    GraphicsPipeline, GraphicsPipelineSettings, ReloadableShader, Shader, VertexBufferLayout,
};
use crate::primitives::CacheMap;
use crate::binding_resources::{BindingResource};

pub struct DropSignal<T> {
    inner: T,
    // Sends a cancellation on drop
    _exit_signal: futures::channel::oneshot::Sender<()>,
}

impl<T> std::ops::Deref for DropSignal<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

struct SizedImage {
    image: Resource<Image>,
    extent: vk::Extent2D,
}

pub(crate) type Allocator = Arc<parking_lot::Mutex<gpu_allocator::vulkan::Allocator>>;

pub struct Device {
    pub inner: ash::Device,
    debug_utils: ext::DebugUtils,
    pub swapchain: khr::Swapchain,
    pub(crate) shaders: CacheMap<&'static str, DropSignal<Arc<ReloadableShader>>>,
    pub(crate) allocator: Allocator,
    pub(crate) graphics_pipelines: CacheMap<&'static str, GraphicsPipeline>,
    pub(crate) descriptor_sets: CacheMap<&'static str, DescriptorSet>,
    images: CacheMap<&'static str, SizedImage>,
    threads: parking_lot::Mutex<Vec<std::thread::JoinHandle<()>>>,
    next_resource_id: std::sync::atomic::AtomicU32,
}

impl Device {
    pub fn new(
        device: ash::Device,
        allocator: gpu_allocator::vulkan::Allocator,
        debug_utils: ext::DebugUtils,
        swapchain: khr::Swapchain,
    ) -> Self {
        Self {
            inner: device,
            debug_utils,
            swapchain,
            allocator: Arc::new(parking_lot::Mutex::new(allocator)),
            shaders: Default::default(),
            graphics_pipelines: Default::default(),
            descriptor_sets: Default::default(),
            next_resource_id: Default::default(),
            images: Default::default(),
            threads: Default::default(),
        }
    }

    pub fn get_image(
        &self,
        name: &'static str,
        extent: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> &Image {
        let image_fn = || SizedImage {
            image: self.create_resource(Image::new_2d(self, name, extent, format, usage)),
            extent,
        };

        let image = self.images.get_or_insert_with(name, image_fn);

        if extent != image.extent {
            &self.images.insert_or_replace(name, image_fn()).image
        } else {
            &image.image
        }
    }

    pub fn get_shader(&self, filename: &'static str) -> &ReloadableShader {
        self.shaders.get_or_insert_with(filename, || {
            let (exit_tx, mut exit_rx) = futures::channel::oneshot::channel::<()>();

            let shader = Arc::new(ReloadableShader {
                shader: parking_lot::Mutex::new(Shader::load(
                    &self.inner,
                    &self.debug_utils,
                    filename,
                )),
                version: Default::default(),
            });

            let handle = std::thread::spawn({
                let shader = shader.clone();
                let device = self.inner.clone();
                let debug_utils = self.debug_utils.clone();
                move || {
                    use notify::Watcher;

                    let (mut tx, mut rx) = futures::channel::mpsc::channel(1);

                    let mut watcher = notify::recommended_watcher(move |res| {
                        futures::executor::block_on(async {
                            tx.send(res).await.unwrap();
                        })
                    })
                    .unwrap();

                    watcher
                        .watch(
                            std::path::Path::new(filename),
                            notify::RecursiveMode::NonRecursive,
                        )
                        .unwrap();

                    loop {
                        match futures::executor::block_on(futures::future::select(
                            rx.next(),
                            &mut exit_rx,
                        )) {
                            futures::future::Either::Left((Some(_), _)) => {
                                println!("Reloading {}", filename);
                                *shader.shader.lock() =
                                    Shader::load(&device, &debug_utils, filename);
                                shader
                                    .version
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            _ => {
                                log::debug!("Dropping {}", filename);
                                break;
                            }
                        }
                    }
                }
            });

            self.threads.lock().push(handle);

            DropSignal {
                inner: shader,
                _exit_signal: exit_tx,
            }
        })
    }

    pub fn get_graphics_pipeline(
        &self,
        name: &'static str,
        vertex_shader: &ReloadableShader,
        fragment_shader: Option<&ReloadableShader>,
        settings: &GraphicsPipelineSettings,
        color_attachment_formats: &[vk::Format],
        vertex_binding_layouts: &[VertexBufferLayout],
    ) -> &GraphicsPipeline {
        let shader_versions = (
            vertex_shader.get_version(),
            fragment_shader.map(|shader| shader.get_version()),
        );

        let pipeline_fn = || {
            GraphicsPipeline::new(
                self,
                name,
                &vertex_shader.shader.lock(),
                fragment_shader.as_ref().map(|shader| shader.shader.lock()),
                settings,
                color_attachment_formats,
                vertex_binding_layouts,
                shader_versions,
            )
        };

        let pipeline = self
            .graphics_pipelines
            .get_or_insert_with(name, pipeline_fn);

        if pipeline.shader_versions != shader_versions {
            self.graphics_pipelines
                .insert_or_replace(name, pipeline_fn())
        } else {
            pipeline
        }
    }

    pub fn flush(&mut self) {
        self.shaders.flush("shaders");
        self.graphics_pipelines.flush("graphics pipelines");
        self.descriptor_sets.flush("descriptor sets");
        self.images.flush("images");
    }

    pub fn create_resource<T>(&self, resource: T) -> Resource<T> {
        Resource {
            inner: resource,
            id: self
                .next_resource_id
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }

    pub fn set_object_name<T: vk::Handle>(&self, handle: T, name: &str) {
        set_object_name(&self.inner, &self.debug_utils, handle, name);
    }

    pub fn get_descriptor_set(
        &self,
        name: &'static str,
        graphics_pipeline: &GraphicsPipeline,
        bindings: &[BindingResource],
    ) -> &DescriptorSet {
        let ids: Vec<_> = bindings.iter().map(|resource| resource.id()).collect();

        let descriptor_set_fn = || {
            DescriptorSet::new(
                self,
                name,
                graphics_pipeline.descriptor_set_layout,
                bindings,
                &ids,
            )
        };

        let descriptor_set = self
            .descriptor_sets
            .get_or_insert_with(name, descriptor_set_fn);

        if ids != descriptor_set.ids {
            self.descriptor_sets
                .insert_or_replace(name, descriptor_set_fn())
        } else {
            descriptor_set
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        drop(std::mem::take(&mut self.shaders));

        let mut threads = self.threads.lock();

        // Ensure that we join the threads and destroy the shaders before destroying the device
        for thread in threads.drain(..) {
            if let Err(err) = thread.join() {
                eprintln!("Error: {:?}", err);
            }
        }
    }
}

pub(crate) fn set_object_name<T: vk::Handle>(
    device: &ash::Device,
    debug_utils: &ext::DebugUtils,
    handle: T,
    name: &str,
) {
    let name = CString::new(name).unwrap();

    unsafe {
        debug_utils
            .debug_utils_set_object_name(
                device.handle(),
                &*vk::DebugUtilsObjectNameInfoEXT::builder()
                    .object_type(T::TYPE)
                    .object_handle(handle.as_raw())
                    .object_name(&name),
            )
            .unwrap();
    }
}
