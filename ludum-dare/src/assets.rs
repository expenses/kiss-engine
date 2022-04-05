use glam::Vec2;
use glam::Vec3;
use kiss_engine_wgpu::{Device, Resource, Texture};
use wgpu::util::DeviceExt;

use gltf_helpers::{NodeTree, animation, Similarity};

pub(crate) struct Model {
    positions: wgpu::Buffer,
    normals: wgpu::Buffer,
    uvs: wgpu::Buffer,
    indices: wgpu::Buffer,
    num_indices: u32,
    pub(crate) joints: wgpu::Buffer,
    pub(crate) weights: wgpu::Buffer,
    pub(crate) depth_first_nodes: Vec<(usize, Option<usize>)>,
    pub(crate) animations: Vec<animation::Animation>,
    pub(crate) inverse_bind_transforms: Vec<Similarity>,
    pub(crate) joint_indices_to_node_indices: Vec<usize>,
}

impl Model {
    pub(crate) fn new(
        bytes: &[u8],
        device: &wgpu::Device,
        name: &str,
    ) -> (Self, animation::AnimationJoints) {
        let gltf = gltf::Gltf::from_slice(bytes).expect("Failed to read gltf");

        let buffer_blob = gltf
            .blob
            .as_ref()
            .expect("Failed to get buffer blob. Make sure you're loading a .glb and not a .gltf.");

        let node_tree = NodeTree::new(gltf.nodes());

        let mut indices = Vec::new();
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut joints = Vec::new();
        let mut weights = Vec::new();

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

                let read_indices = reader
                    .read_indices()
                    .expect("Failed to read indices")
                    .into_u32();

                let num_existing_vertices = positions.len();

                indices.extend(read_indices.map(|index| index + num_existing_vertices as u32));

                positions.extend(
                    reader
                        .read_positions()
                        .expect("Failed to read positions")
                        .map(|pos| transform * Vec3::from(pos)),
                );

                normals.extend(
                    reader
                        .read_normals()
                        .expect("Failed to read normals")
                        .map(|normal| transform.rotation * Vec3::from(normal)),
                );

                uvs.extend(
                    reader
                        .read_tex_coords(0)
                        .expect("Failed to read tex coords (0)")
                        .into_f32()
                        .map(Vec2::from),
                );

                if let Some(read_joints) = reader.read_joints(0) {
                    joints.extend(read_joints.into_u16());
                }

                if let Some(read_weights) = reader.read_weights(0) {
                    weights.extend(read_weights.into_f32());
                }
            }
        }

        let depth_first_nodes: Vec<_> = node_tree.iter_depth_first().collect();
        let animations = animation::read_animations(gltf.animations(), buffer_blob, name);
        let animation_joints = animation::AnimationJoints::new(gltf.nodes(), &depth_first_nodes);

        let skin = gltf.skins().next();

        let joint_indices_to_node_indices: Vec<usize> = if let Some(skin) = skin.as_ref() {
            skin.joints().map(|node| node.index()).collect()
        } else {
            gltf.nodes().map(|node| node.index()).collect()
        };

        let inverse_bind_transforms: Vec<Similarity> = if let Some(skin) = skin.as_ref() {
            skin.reader(|buffer| {
                assert_eq!(buffer.index(), 0);
                Some(buffer_blob)
            })
            .read_inverse_bind_matrices()
            .expect("Missing inverse bind matrices")
            .map(|matrix| {
                let (translation, rotation, scale) = gltf::scene::Transform::Matrix { matrix }.decomposed();
                Similarity::new_from_gltf(translation, rotation, scale)
            })
            .collect()
        } else {
            gltf.nodes()
                .map(|node| node_tree.transform_of(node.index()).inverse())
                .collect()
        };

        (
            Self {
                positions: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{} positions buffer", name)),
                    contents: bytemuck::cast_slice(&positions),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{} indices buffer", name)),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
                normals: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{} normals buffer", name)),
                    contents: bytemuck::cast_slice(&normals),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                uvs: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{} uvs buffer", name)),
                    contents: bytemuck::cast_slice(&uvs),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                joints: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{} joints buffer", name)),
                    contents: bytemuck::cast_slice(&joints),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                weights: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{} weights buffer", name)),
                    contents: bytemuck::cast_slice(&weights),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                num_indices: indices.len() as u32,
                depth_first_nodes,
                animations,
                joint_indices_to_node_indices,
                inverse_bind_transforms,
            },
            animation_joints,
        )
    }

    pub(crate) fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, num_instances: u32) {
        render_pass.set_vertex_buffer(0, self.positions.slice(..));
        render_pass.set_vertex_buffer(1, self.normals.slice(..));
        render_pass.set_vertex_buffer(2, self.uvs.slice(..));
        render_pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);

        render_pass.draw_indexed(0..self.num_indices, 0, 0..num_instances);
    }
}

pub(crate) fn load_image(
    device: &Device,
    queue: &wgpu::Queue,
    bytes: &[u8],
    name: &str,
) -> Resource<Texture> {
    let image = image::load_from_memory(bytes)
        .expect("Failed to read image")
        .to_rgba8();

    let texture = device.inner.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some(name),
            size: wgpu::Extent3d {
                width: image.width(),
                height: image.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
        },
        &*image,
    );

    device.create_resource(Texture {
        view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        texture,
    })
}
