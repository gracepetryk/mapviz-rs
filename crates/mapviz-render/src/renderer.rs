//! Concrete wgpu renderer for the 2D primitive set.

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use mapviz_core::{
    Camera2d, FillVertex, Frame, LineInstance, QuadInstance, TexturedQuad, tessellate_shape,
};
use wgpu::util::DeviceExt;

/// Errors raised while setting up or driving the renderer.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    /// No GPU adapter could satisfy the request (no WebGPU support?).
    #[error("no suitable GPU adapter found")]
    NoAdapter,
    /// Failed to create the rendering surface from the target.
    #[error("failed to create surface: {0}")]
    CreateSurface(#[from] wgpu::CreateSurfaceError),
    /// Failed to acquire a logical device and queue.
    #[error("failed to request device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
}

/// Camera data as laid out for the shader uniform.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

// GPU-layout mirrors of the core primitives. Each is `#[repr(C)]` + `Pod` so it
// can be uploaded directly; the matching core type stays free of GPU concerns,
// and a `From` keeps the `bytemuck` casts in this crate.

/// GPU layout for a quad instance. Mirrors [`QuadInstance`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuad {
    center: [f32; 2],
    half_extent: [f32; 2],
    color: [f32; 4],
}

impl From<&QuadInstance> for GpuQuad {
    fn from(q: &QuadInstance) -> Self {
        Self {
            center: q.center,
            half_extent: q.half_extent,
            color: q.color,
        }
    }
}

/// GPU layout for a line instance. Mirrors [`LineInstance`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLine {
    start: [f32; 2],
    end: [f32; 2],
    width: f32,
    color: [f32; 4],
}

impl From<&LineInstance> for GpuLine {
    fn from(l: &LineInstance) -> Self {
        Self {
            start: l.start,
            end: l.end,
            width: l.width,
            color: l.color,
        }
    }
}

/// GPU layout for a fill vertex. Mirrors [`FillVertex`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuFillVertex {
    position: [f32; 2],
    color: [f32; 4],
}

impl From<&FillVertex> for GpuFillVertex {
    fn from(v: &FillVertex) -> Self {
        Self {
            position: v.position,
            color: v.color,
        }
    }
}

/// GPU layout for a textured-quad instance. Mirrors the `center`/`half_extent`
/// of a [`TexturedQuad`]; the texture itself is bound separately, not packed
/// here. UVs are derived from the vertex index in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTexQuad {
    center: [f32; 2],
    half_extent: [f32; 2],
}

impl From<&TexturedQuad> for GpuTexQuad {
    fn from(q: &TexturedQuad) -> Self {
        Self {
            center: q.center,
            half_extent: q.half_extent,
        }
    }
}

/// How one GPU instance type is rendered: its shader, topology, vertex layout,
/// and vertices-per-instance. Implementing this is all a new *instanced*
/// primitive needs to get a pipeline (via [`build_pipeline`]) and buffering +
/// draws (via [`InstancedBatch`]). Non-instanced draw models (e.g. indexed
/// meshes) will get their own holder rather than this trait.
///
/// Several items are only read from `build_pipeline`, which is reached only on
/// the web (the surface is canvas-bound), so they read as dead code on native
/// builds — hence the target-gated `allow`.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
trait GpuInstance: Pod + Zeroable {
    /// Debug label for the pipeline and buffer.
    const LABEL: &'static str;
    /// WGSL source, with `vs_main` / `fs_main` entry points.
    const SHADER: &'static str;
    /// Topology the vertex shader emits.
    const TOPOLOGY: wgpu::PrimitiveTopology;
    /// Vertices emitted per instance (e.g. 4 for a triangle-strip quad).
    const VERTICES: u32;
    /// Per-instance vertex buffer layout.
    fn vertex_layout() -> wgpu::VertexBufferLayout<'static>;
}

impl GpuInstance for GpuQuad {
    const LABEL: &'static str = "quad";
    const SHADER: &'static str = include_str!("quad.wgsl");
    const TOPOLOGY: wgpu::PrimitiveTopology = wgpu::PrimitiveTopology::TriangleStrip;
    const VERTICES: u32 = 4;

    fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: [wgpu::VertexAttribute; 3] =
            wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuQuad>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &ATTRS,
        }
    }
}

impl GpuInstance for GpuLine {
    const LABEL: &'static str = "line";
    const SHADER: &'static str = include_str!("line.wgsl");
    const TOPOLOGY: wgpu::PrimitiveTopology = wgpu::PrimitiveTopology::TriangleStrip;
    const VERTICES: u32 = 4;

    fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: [wgpu::VertexAttribute; 4] =
            wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32, 3 => Float32x4];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuLine>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &ATTRS,
        }
    }
}

/// Build a render pipeline for an instanced primitive type, sharing the camera
/// bind group layout. All the boilerplate that doesn't vary per primitive lives
/// here; what varies (shader, vertex layout, topology) comes from `T`.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn build_pipeline<T: GpuInstance>(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    camera_bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(T::LABEL),
        source: wgpu::ShaderSource::Wgsl(T::SHADER.into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(T::LABEL),
        bind_group_layouts: &[Some(camera_bgl)],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(T::LABEL),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[T::vertex_layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: T::TOPOLOGY,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// A pipeline plus a growable instance buffer for one GPU instance type.
/// Accumulates instances for a frame (`begin`/`push`), uploads them in a single
/// write (`upload`), then issues per-batch draws over ranges of that buffer.
struct InstancedBatch<T: GpuInstance> {
    pipeline: wgpu::RenderPipeline,
    buffer: wgpu::Buffer,
    /// Buffer capacity in instances.
    capacity: u32,
    /// Instances accumulated this frame, reused across frames.
    scratch: Vec<T>,
}

impl<T: GpuInstance> InstancedBatch<T> {
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let pipeline = build_pipeline::<T>(device, format, camera_bgl);
        // A one-instance placeholder; grows on first upload that needs it.
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(T::LABEL),
            size: std::mem::size_of::<T>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            buffer,
            capacity: 0,
            scratch: Vec::new(),
        }
    }

    /// Drop the previous frame's instances, keeping capacity.
    fn begin(&mut self) {
        self.scratch.clear();
    }

    /// Append a batch of instances, returning its `(offset, count)` within this
    /// frame's buffer so it can later be drawn as a contiguous range.
    fn push(&mut self, instances: impl IntoIterator<Item = T>) -> (u32, u32) {
        let offset = self.scratch.len() as u32;
        self.scratch.extend(instances);
        (offset, self.scratch.len() as u32 - offset)
    }

    /// Upload all accumulated instances in one write, growing the buffer if needed.
    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let len = self.scratch.len() as u32;
        if len == 0 {
            return;
        }
        if len > self.capacity {
            self.buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(T::LABEL),
                contents: bytemuck::cast_slice(&self.scratch),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            self.capacity = len;
        } else {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.scratch));
        }
    }

    /// Draw one previously-pushed range as an instanced pass.
    fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        camera_bind_group: &wgpu::BindGroup,
        offset: u32,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, camera_bind_group, &[]);
        pass.set_vertex_buffer(0, self.buffer.slice(..));
        pass.draw(0..T::VERTICES, offset..offset + count);
    }
}

/// Holder for an indexed triangle-mesh draw — separate from [`InstancedBatch`]
/// because the draw model is different (vertex buffer + index buffer, no
/// instancing). Accumulates all `Mesh` batches for a frame (`begin`/`push`),
/// concatenates them into a single vertex and index buffer pair (`upload`), and
/// draws each over its sub-range of that buffer.
struct IndexedBatch {
    pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    /// Buffer capacity in vertices.
    vertex_capacity: u32,
    /// Buffer capacity in indices.
    index_capacity: u32,
    /// Vertices accumulated this frame.
    vertices: Vec<GpuFillVertex>,
    /// Indices accumulated this frame.
    indices: Vec<u32>,
}

impl IndexedBatch {
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh"),
            source: wgpu::ShaderSource::Wgsl(include_str!("mesh.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh"),
            bind_group_layouts: &[Some(camera_bgl)],
            immediate_size: 0,
        });
        const ATTRS: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4];
        let pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mesh"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<GpuFillVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &ATTRS,
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        // One-element placeholder buffers that grow on demand.
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh vertices"),
            size: std::mem::size_of::<GpuFillVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh indices"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            vertex_buf,
            index_buf,
            vertex_capacity: 0,
            index_capacity: 0,
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Drop the previous frame's data, keeping allocations.
    fn begin(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    /// Append one mesh batch. Returns `(base_vertex, base_index, index_count)`
    /// so the caller can record a draw command for this sub-range.
    ///
    /// `base_vertex` is an offset added to every index before indexing the
    /// vertex buffer, so each batch's indices stay local `0..N`.
    fn push(
        &mut self,
        mesh_verts: &[FillVertex],
        mesh_indices: &[u32],
    ) -> (i32, u32, u32) {
        let base_vertex = self.vertices.len() as i32;
        let base_index = self.indices.len() as u32;
        self.vertices.extend(mesh_verts.iter().map(GpuFillVertex::from));
        self.indices.extend_from_slice(mesh_indices);
        (base_vertex, base_index, mesh_indices.len() as u32)
    }

    /// Upload both buffers in a single write each, growing if needed.
    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.vertices.is_empty() {
            return;
        }

        let vlen = self.vertices.len() as u32;
        if vlen > self.vertex_capacity {
            self.vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh vertices"),
                contents: bytemuck::cast_slice(&self.vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            self.vertex_capacity = vlen;
        } else {
            queue.write_buffer(
                &self.vertex_buf,
                0,
                bytemuck::cast_slice(&self.vertices),
            );
        }

        let ilen = self.indices.len() as u32;
        if ilen > self.index_capacity {
            self.index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh indices"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            });
            self.index_capacity = ilen;
        } else {
            queue.write_buffer(
                &self.index_buf,
                0,
                bytemuck::cast_slice(&self.indices),
            );
        }
    }

    /// Draw one previously-pushed mesh sub-range.
    fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        camera_bind_group: &wgpu::BindGroup,
        base_vertex: i32,
        base_index: u32,
        index_count: u32,
    ) {
        if index_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, camera_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(
            base_index..base_index + index_count,
            base_vertex,
            0..1,
        );
    }
}

/// A GPU texture uploaded from a [`TextureHandle`], plus the bind group that
/// binds it (and the shared sampler) for the textured-quad pipeline. The
/// `_image` keeps the source handle alive so its pointer — used as the cache
/// key — stays unique for the texture's lifetime.
struct CachedTexture {
    bind_group: wgpu::BindGroup,
    _image: mapviz_core::TextureHandle,
}

/// A pipeline plus instance buffer for textured quads, with a cache of uploaded
/// GPU textures keyed by source-image identity.
///
/// Unlike [`InstancedBatch`], textured quads can't all be drawn in one
/// instanced call: each may sample a different texture, so each is its own draw
/// with its own texture bind group. Instances still share one buffer; draws
/// index into it one quad at a time.
struct TexturedBatch {
    pipeline: wgpu::RenderPipeline,
    /// Layout for the per-texture bind group (texture view + sampler).
    texture_bgl: wgpu::BindGroupLayout,
    /// Sampler shared by every textured quad.
    sampler: wgpu::Sampler,
    buffer: wgpu::Buffer,
    /// Buffer capacity in instances.
    capacity: u32,
    /// Instances accumulated this frame.
    scratch: Vec<GpuTexQuad>,
    /// Uploaded textures, keyed by the source handle's pointer. Persists across
    /// frames so a tile's pixels upload once.
    cache: HashMap<usize, CachedTexture>,
}

impl TexturedBatch {
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tex"),
            source: wgpu::ShaderSource::Wgsl(include_str!("tex.wgsl").into()),
        });

        let texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tex bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tex sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tex"),
            bind_group_layouts: &[Some(camera_bgl), Some(&texture_bgl)],
            immediate_size: 0,
        });

        const ATTRS: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tex"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuTexQuad>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &ATTRS,
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tex"),
            size: std::mem::size_of::<GpuTexQuad>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            texture_bgl,
            sampler,
            buffer,
            capacity: 0,
            scratch: Vec::new(),
            cache: HashMap::new(),
        }
    }

    /// Drop the previous frame's instances, keeping the texture cache.
    fn begin(&mut self) {
        self.scratch.clear();
    }

    /// Ensure `quad`'s texture is uploaded, then append its instance. Returns
    /// `(instance_index, texture_key)` to record a single-quad draw.
    fn push(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        quad: &TexturedQuad,
    ) -> (u32, usize) {
        let key = Arc::as_ptr(&quad.texture) as usize;
        self.cache
            .entry(key)
            .or_insert_with(|| upload_texture(device, queue, &self.texture_bgl, &self.sampler, &quad.texture));
        let index = self.scratch.len() as u32;
        self.scratch.push(GpuTexQuad::from(quad));
        (index, key)
    }

    /// Upload accumulated instances in one write, growing the buffer if needed.
    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let len = self.scratch.len() as u32;
        if len == 0 {
            return;
        }
        if len > self.capacity {
            self.buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tex"),
                contents: bytemuck::cast_slice(&self.scratch),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            self.capacity = len;
        } else {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.scratch));
        }
    }

    /// Draw one previously-pushed quad with its texture bound.
    fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        camera_bind_group: &wgpu::BindGroup,
        instance_index: u32,
        texture_key: usize,
    ) {
        let Some(entry) = self.cache.get(&texture_key) else {
            return;
        };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, camera_bind_group, &[]);
        pass.set_bind_group(1, &entry.bind_group, &[]);
        pass.set_vertex_buffer(0, self.buffer.slice(..));
        pass.draw(0..4, instance_index..instance_index + 1);
    }
}

/// Upload a [`TextureHandle`]'s pixels to a new GPU texture and build its bind
/// group. A free function (not a method) so it can borrow the batch's layout and
/// sampler while `push` mutably borrows the cache.
fn upload_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    image: &mapviz_core::TextureHandle,
) -> CachedTexture {
    let size = wgpu::Extent3d {
        width: image.width.max(1),
        height: image.height.max(1),
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tile texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // sRGB so sampling decodes to linear; the surface re-encodes on output.
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &image.rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * image.width),
            rows_per_image: Some(image.height),
        },
        size,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tile texture bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    CachedTexture {
        bind_group,
        _image: image.clone(),
    }
}

/// One entry in the frame's draw order: which batch, and which instance range.
#[derive(Clone, Copy)]
enum DrawCmd {
    Quads(u32, u32),
    Lines(u32, u32),
    /// Indexed mesh: base_vertex offset, index buffer start, index count.
    Mesh(i32, u32, u32),
    /// Textured quad: instance index, texture cache key.
    TexturedQuad(u32, usize),
}

/// A concrete wgpu renderer that draws a [`Frame`] under a [`Camera2d`].
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    quads: InstancedBatch<GpuQuad>,
    lines: InstancedBatch<GpuLine>,
    meshes: IndexedBatch,
    textured: TexturedBatch,
    /// Per-frame draw order, reused across frames.
    draw_order: Vec<DrawCmd>,
}

impl Renderer {
    /// Assemble the renderer's GPU resources. Shared by all targets so the
    /// resource code is type-checked on native builds even though the surface
    /// itself is only created on the web (see [`Renderer::new`]).
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    fn from_parts(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
    ) -> Self {
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let quads = InstancedBatch::new(&device, config.format, &camera_bgl);
        let lines = InstancedBatch::new(&device, config.format, &camera_bgl);
        let meshes = IndexedBatch::new(&device, config.format, &camera_bgl);
        let textured = TexturedBatch::new(&device, config.format, &camera_bgl);

        Self {
            device,
            queue,
            surface,
            config,
            camera_buffer,
            camera_bind_group,
            quads,
            lines,
            meshes,
            textured,
            draw_order: Vec::new(),
        }
    }

    /// Reconfigure the surface for a new size in physical pixels.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    /// Render a frame under the given camera.
    pub fn render(&mut self, camera: &Camera2d, frame: &Frame) -> Result<(), RenderError> {
        // Update the camera uniform.
        let uniform = CameraUniform {
            view_proj: camera.view_proj().to_cols_array_2d(),
        };
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));

        // Bucket each batch into its per-type buffer, recording the draw order
        // and each batch's instance range. Batches keep submission order, so
        // layer order is render (painter's) order even across primitive kinds.
        self.quads.begin();
        self.lines.begin();
        self.meshes.begin();
        self.textured.begin();
        self.draw_order.clear();
        // Tessellate each shape into draw instances, bucketed by draw model.
        // Within a shape we record fill (under), then texture, then stroke, then
        // markers (on top); across shapes, submission order is render (painter's)
        // order.
        for shape in &frame.shapes {
            let data = tessellate_shape(shape);
            if !data.fill_indices.is_empty() {
                let (base_vertex, base_index, index_count) =
                    self.meshes.push(&data.fill_vertices, &data.fill_indices);
                self.draw_order
                    .push(DrawCmd::Mesh(base_vertex, base_index, index_count));
            }
            for quad in &data.textured_quads {
                let (index, key) = self.textured.push(&self.device, &self.queue, quad);
                self.draw_order.push(DrawCmd::TexturedQuad(index, key));
            }
            if !data.strokes.is_empty() {
                let (offset, count) = self.lines.push(data.strokes.iter().map(GpuLine::from));
                self.draw_order.push(DrawCmd::Lines(offset, count));
            }
            if !data.markers.is_empty() {
                let (offset, count) = self.quads.push(data.markers.iter().map(GpuQuad::from));
                self.draw_order.push(DrawCmd::Quads(offset, count));
            }
        }
        self.quads.upload(&self.device, &self.queue);
        self.lines.upload(&self.device, &self.queue);
        self.meshes.upload(&self.device, &self.queue);
        self.textured.upload(&self.device, &self.queue);

        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                // Surface fell out of sync; reconfigure and skip this frame.
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            // Timeout / Occluded / Validation: skip this frame and try again.
            _ => return Ok(()),
        };
        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.07,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            for cmd in &self.draw_order {
                match *cmd {
                    DrawCmd::Quads(offset, count) => {
                        self.quads
                            .draw(&mut pass, &self.camera_bind_group, offset, count);
                    }
                    DrawCmd::Lines(offset, count) => {
                        self.lines
                            .draw(&mut pass, &self.camera_bind_group, offset, count);
                    }
                    DrawCmd::TexturedQuad(index, key) => {
                        self.textured
                            .draw(&mut pass, &self.camera_bind_group, index, key);
                    }
                    DrawCmd::Mesh(base_vertex, base_index, index_count) => {
                        self.meshes.draw(
                            &mut pass,
                            &self.camera_bind_group,
                            base_vertex,
                            base_index,
                            index_count,
                        );
                    }
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
impl Renderer {
    /// Create a renderer that draws into the given canvas, sized in physical
    /// pixels. Async because adapter and device acquisition are async.
    pub async fn new(
        canvas: web_sys::HtmlCanvasElement,
        width: u32,
        height: u32,
    ) -> Result<Self, RenderError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });

        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .map_err(|_| RenderError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("mapviz device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        Ok(Self::from_parts(device, queue, surface, config))
    }
}
