use crate::device::{Allocator, Device};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc};

#[derive(Debug)]
pub struct Resource<T> {
    pub inner: T,
    pub(crate) id: u32,
}

impl<T> std::ops::Deref for Resource<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct Buffer {
    device: ash::Device,
    allocator: Allocator,
    allocation: Option<Allocation>,
    pub buffer: vk::Buffer,
}

impl Buffer {
    pub fn new_from_bytes(
        device: &Device,
        bytes: &[u8],
        name: &str,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        let buffer_size = bytes.len() as vk::DeviceSize;

        let buffer = unsafe {
            device.inner.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(buffer_size)
                    .usage(usage),
                None,
            )
        }
        .unwrap();

        let requirements = unsafe { device.inner.get_buffer_memory_requirements(buffer) };

        let mut allocation = device
            .allocator
            .lock()
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
            })
            .unwrap();

        let slice = allocation.mapped_slice_mut().unwrap();

        slice[..bytes.len()].copy_from_slice(bytes);

        unsafe {
            device
                .inner
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .unwrap();
        };

        device.set_object_name(buffer, name);

        Self {
            device: device.inner.clone(),
            allocator: device.allocator.clone(),
            buffer,
            allocation: Some(allocation),
        }
    }

    pub fn write_mapped(&mut self, bytes: &[u8], offset: usize) {
        let slice = self
            .allocation
            .as_mut()
            .unwrap()
            .mapped_slice_mut()
            .ok_or("Attempted to write to a buffer that wasn't mapped")
            .unwrap();
        slice[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    pub(crate) fn as_descriptor_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.buffer,
            range: vk::WHOLE_SIZE,
            offset: 0,
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
        }

        if let Some(allocation) = self.allocation.take() {
            if let Err(err) = self.allocator.lock().free(allocation) {
                eprintln!("Drop error: {}", err);
            }
        }
    }
}

pub struct Image {
    device: ash::Device,
    allocator: Allocator,
    allocation: Option<Allocation>,
    pub image: vk::Image,
    pub view: vk::ImageView,
}

impl Image {
    pub(crate) fn new_2d(
        device: &Device,
        name: &'static str,
        extent: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Self {
        let image = unsafe {
            device.inner.create_image(
                &vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .usage(usage),
                None,
            )
        }
        .unwrap();

        let requirements = unsafe { device.inner.get_image_memory_requirements(image) };

        let allocation = device
            .allocator
            .lock()
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: false,
            })
            .unwrap();

        unsafe {
            device
                .inner
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .unwrap();
        };

        let subresource_range = *vk::ImageSubresourceRange::builder()
            .aspect_mask(if format == vk::Format::D32_SFLOAT {
                vk::ImageAspectFlags::DEPTH
            } else {
                vk::ImageAspectFlags::COLOR
            })
            .level_count(1)
            .layer_count(1);

        let view = unsafe {
            device.inner.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(subresource_range),
                None,
            )
        }
        .unwrap();

        device.set_object_name(image, name);
        device.set_object_name(view, &format!("{} view", name));
        device.set_object_name(unsafe { allocation.memory() }, &format!("{} memory", name));

        Self {
            device: device.inner.clone(),
            allocator: device.allocator.clone(),
            image,
            allocation: Some(allocation),
            view,
        }
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
        }

        if let Some(allocation) = self.allocation.take() {
            if let Err(err) = self.allocator.lock().free(allocation) {
                eprintln!("Drop error: {}", err);
            }
        }
    }
}

pub enum BindingResource<'a> {
    Buffer(&'a Resource<Buffer>),
    Sampler(&'a Resource<vk::Sampler>),
    //Texture(&'a Resource<wgpu::TextureView>),
}

impl<'a> BindingResource<'a> {
    pub(crate) fn id(&self) -> u32 {
        match self {
            Self::Buffer(res) => res.id,
            Self::Sampler(res) => res.id,
            //Self::Texture(res) => res.id,
        }
    }
}

pub struct DescriptorSet {
    device: ash::Device,
    pool: vk::DescriptorPool,
    pub inner: vk::DescriptorSet,
    pub(crate) ids: Vec<u32>,
}

impl DescriptorSet {
    pub(crate) fn new(
        device: &Device,
        name: &str,
        layout: vk::DescriptorSetLayout,
        bindings: &[BindingResource],
        ids: &[u32],
    ) -> Self {
        let mut num_uniform_buffers = 0;
        let mut num_samplers = 0;

        for binding in bindings {
            match binding {
                BindingResource::Sampler(_) => num_samplers += 1,
                BindingResource::Buffer(_) => num_uniform_buffers += 1,
            }
        }

        let mut pool_sizes = vec![
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: num_samplers,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: num_uniform_buffers,
            },
        ];

        pool_sizes.retain(|size| size.descriptor_count > 0);

        let descriptor_pool = unsafe {
            device
                .inner
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::builder()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .unwrap()
        };

        device.set_object_name(descriptor_pool, &format!("{} descriptor pool", name));

        let descriptor_set = unsafe {
            device.inner.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[layout]),
            )
        }
        .unwrap()[0];

        #[derive(Debug)]
        struct OwnedDescriptorWrite {
            ty: vk::DescriptorType,
            info: DescriptorWriteInfo,
        }

        #[derive(Debug)]
        enum DescriptorWriteInfo {
            ImageInfo([vk::DescriptorImageInfo; 1]),
            BufferInfo([vk::DescriptorBufferInfo; 1]),
        }

        let owned_writes: Vec<_> = bindings
            .iter()
            .map(|binding| match binding {
                BindingResource::Sampler(sampler) => OwnedDescriptorWrite {
                    ty: vk::DescriptorType::SAMPLER,
                    info: DescriptorWriteInfo::ImageInfo([
                        *vk::DescriptorImageInfo::builder().sampler(sampler.inner)
                    ]),
                },
                BindingResource::Buffer(buffer) => OwnedDescriptorWrite {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    info: DescriptorWriteInfo::BufferInfo([buffer.as_descriptor_info()]),
                },
            })
            .collect();

        let descriptor_writes: Vec<_> = owned_writes
            .iter()
            .enumerate()
            .map(|(id, owned_write)| {
                let mut write = vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(id as u32)
                    .descriptor_type(owned_write.ty);

                // This & is important!
                match &owned_write.info {
                    DescriptorWriteInfo::ImageInfo(info) => {
                        write = write.image_info(info);
                    }
                    DescriptorWriteInfo::BufferInfo(info) => {
                        write = write.buffer_info(info);
                    }
                }

                *write
            })
            .collect();

        unsafe {
            device.inner.update_descriptor_sets(&descriptor_writes, &[]);
        }

        Self {
            device: device.inner.clone(),
            pool: descriptor_pool,
            inner: descriptor_set,
            ids: ids.to_vec(),
        }
    }
}

impl Drop for DescriptorSet {
    fn drop(&mut self) {
        unsafe { self.device.destroy_descriptor_pool(self.pool, None) }
    }
}
