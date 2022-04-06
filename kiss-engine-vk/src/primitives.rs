use crate::device::Device;
use ash::extensions::khr;
use ash::vk;
use std::ffi::CStr;
use std::os::raw::c_char;

/// A list of C strings and their associated pointers
pub struct CStrList<'a> {
    pub list: Vec<&'a CStr>,
    pointers: Vec<*const c_char>,
}

impl<'a> CStrList<'a> {
    pub fn new(list: Vec<&'a CStr>) -> Self {
        let pointers = list.iter().map(|cstr| cstr.as_ptr()).collect();

        Self { list, pointers }
    }

    pub fn pointers(&self) -> &[*const c_char] {
        &self.pointers
    }
}

pub(crate) struct CacheMap<K, V> {
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
    pub(crate) fn get_or_insert_with<F: FnOnce() -> V>(&self, key: K, func: F) -> &V {
        // Check inserting first, in case we're updating something.
        if let Some(value) = self.inserting.get(&key) {
            value
        } else if let Some(value) = self.map.get(&key) {
            value
        } else {
            self.inserting.insert(key, Box::new(func()))
        }
    }

    pub(crate) fn insert_or_replace(&self, key: K, value: V) -> &V {
        self.inserting.insert(key, Box::new(value))
    }

    pub(crate) fn flush(&mut self, name: &str) {
        let inserting = std::mem::take(&mut self.inserting);

        let old_len = self.map.len();
        let mut num_updating = 0;

        for (key, value) in inserting.into_map().into_iter() {
            self.map.insert(key, value);
            num_updating += 1;
        }

        let new_len = self.map.len();

        if new_len != old_len {
            println!(
                "Adding {} item(s) to {} ({} -> {})",
                new_len - old_len,
                name,
                old_len,
                new_len
            );
        } else if num_updating > 0 {
            println!("Updating {} items in {} ({})", num_updating, name, new_len);
        }
    }
}

/// A callback for the [Vulkan Debug Utils Messenger](https://docs.rs/ash/0.33.3+1.2.191/ash/vk/struct.DebugUtilsMessengerEXT.html)
///
/// # Safety
///
/// Don't use this in any way except as an input to [`DebugUtilsMessengerCreateInfoEXTBuilder.pfn_user_callback`](https://docs.rs/ash/0.33.3+1.2.191/ash/vk/struct.DebugUtilsMessengerCreateInfoEXTBuilder.html#method.pfn_user_callback).
///
#[allow(clippy::nonminimal_bool)]
pub unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let filter_out = (message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
        && message_type == vk::DebugUtilsMessageTypeFlagsEXT::GENERAL)
        || (message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            && message_type == vk::DebugUtilsMessageTypeFlagsEXT::GENERAL)
        || (message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            && message_type == vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION);

    let level = if filter_out {
        log::Level::Trace
    } else {
        match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::Level::Debug,
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::Level::Info,
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::Level::Warn,
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::Level::Error,
            _ => log::Level::Info,
        }
    };

    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    let ty = format!("{:?}", message_type).to_lowercase();
    log::log!(level, "[Debug Msg][{}] {:?}", ty, message);
    vk::FALSE
}

pub struct Swapchain {
    device: ash::Device,
    swapchain_ext: khr::Swapchain,
    pub inner: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
}

impl Swapchain {
    pub fn new(device: &Device, info: &vk::SwapchainCreateInfoKHR) -> Self {
        let swapchain = unsafe { device.swapchain.create_swapchain(info, None).unwrap() };

        let images = unsafe { device.swapchain.get_swapchain_images(swapchain) }.unwrap();

        let views: Vec<_> = unsafe {
            images
                .iter()
                .map(|swapchain_image| {
                    let image_view_info = vk::ImageViewCreateInfo::builder()
                        .image(*swapchain_image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(info.image_format)
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1)
                                .build(),
                        );
                    device.inner.create_image_view(&image_view_info, None)
                })
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };

        Self {
            device: device.inner.clone(),
            swapchain_ext: device.swapchain.clone(),
            inner: swapchain,
            images,
            views,
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        for view in &self.views {
            unsafe {
                self.device.destroy_image_view(*view, None);
            }
        }

        unsafe {
            self.swapchain_ext.destroy_swapchain(self.inner, None);
        }
    }
}
