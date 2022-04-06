use ash::vk;
use glam::Vec2;
use glam::Vec3;
use kiss_engine_vk::Buffer;
use kiss_engine_vk::{binding_resources::Resource, device::Device};

use gltf_helpers::{animation, NodeTree, Similarity};

pub(crate) struct Model {
    pub(crate) positions: Resource<Buffer>,
    pub(crate) normals: Resource<Buffer>,
    pub(crate) uvs: Resource<Buffer>,
    pub(crate) indices: Resource<Buffer>,
    pub(crate) num_indices: u32,
    pub(crate) joints: Option<Resource<Buffer>>,
    pub(crate) weights: Option<Resource<Buffer>>,
    pub(crate) depth_first_nodes: Vec<(usize, Option<usize>)>,
    pub(crate) animations: Vec<animation::Animation>,
    pub(crate) inverse_bind_transforms: Vec<Similarity>,
    pub(crate) joint_indices_to_node_indices: Vec<usize>,
}

impl Model {
    pub(crate) fn new(
        bytes: &[u8],
        device: &Device,
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
                let (translation, rotation, scale) =
                    gltf::scene::Transform::Matrix { matrix }.decomposed();
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
                positions: device.create_buffer(
                    &format!("{} positions buffer", name),
                    bytemuck::cast_slice(&positions),
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                ),
                indices: device.create_buffer(
                    &format!("{} indices buffer", name),
                    bytemuck::cast_slice(&indices),
                    vk::BufferUsageFlags::INDEX_BUFFER,
                ),
                normals: device.create_buffer(
                    &format!("{} normals buffer", name),
                    bytemuck::cast_slice(&normals),
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                ),
                uvs: device.create_buffer(
                    &format!("{} uvs buffer", name),
                    bytemuck::cast_slice(&uvs),
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                ),
                joints: if !joints.is_empty() {
                    Some(device.create_buffer(
                        &format!("{} joints buffer", name),
                        bytemuck::cast_slice(&joints),
                        vk::BufferUsageFlags::VERTEX_BUFFER,
                    ))
                } else {
                    None
                },
                weights: if !weights.is_empty() {
                    Some(device.create_buffer(
                        &format!("{} weights buffer", name),
                        bytemuck::cast_slice(&weights),
                        vk::BufferUsageFlags::VERTEX_BUFFER,
                    ))
                } else {
                    None
                },
                num_indices: indices.len() as u32,
                depth_first_nodes,
                animations,
                joint_indices_to_node_indices,
                inverse_bind_transforms,
            },
            animation_joints,
        )
    }

    pub(crate) unsafe fn render(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        num_instances: u32,
    ) {
        device.cmd_bind_vertex_buffers(
            command_buffer,
            0,
            &[self.positions.buffer, self.normals.buffer, self.uvs.buffer],
            &[0, 0, 0],
        );
        device.cmd_bind_index_buffer(
            command_buffer,
            self.indices.buffer,
            0,
            vk::IndexType::UINT32,
        );
        device.cmd_draw_indexed(command_buffer, self.num_indices, num_instances, 0, 0, 0);
    }

    pub(crate) unsafe fn render_animated(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        num_instances: u32,
    ) {
        device.cmd_bind_vertex_buffers(
            command_buffer,
            0,
            &[
                self.positions.buffer,
                self.normals.buffer,
                self.uvs.buffer,
                self.joints.as_ref().unwrap().buffer,
                self.weights.as_ref().unwrap().buffer,
            ],
            &[0, 0, 0, 0, 0],
        );
        device.cmd_bind_index_buffer(
            command_buffer,
            self.indices.buffer,
            0,
            vk::IndexType::UINT32,
        );
        device.cmd_draw_indexed(command_buffer, self.num_indices, num_instances, 0, 0, 0);
    }
}

pub fn load_image<'a>(
    device: &'a Device,
    queue: vk::Queue,
    bytes: &[u8],
    name: &'static str,
) -> kiss_engine_vk::Resource<kiss_engine_vk::Image> {
    let image = image::load_from_memory(bytes)
        .expect("Failed to read image")
        .to_rgba8();

    kiss_engine_vk::binding_resources::load_image(device, queue, &*image, image.width(), image.height(), name)
}
