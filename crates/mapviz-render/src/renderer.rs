//! Concrete wgpu renderer for the 2D primitive set.

use bytemuck::{Pod, Zeroable};
use mapviz_core::{Camera2d, Frame, Primitive, QuadInstance};
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

/// A quad instance as laid out in the GPU instance buffer. Mirrors
/// [`QuadInstance`] (same field order/types) but carries the `Pod`/`Zeroable`
/// impls that keep `bytemuck` casts in this crate rather than in core.
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

/// A concrete wgpu renderer that draws a [`Frame`] under a [`Camera2d`].
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    /// Instance buffer and its current capacity in instances.
    instance_buffer: wgpu::Buffer,
    instance_capacity: u32,
    /// Scratch buffer reused each frame to gather GPU instances.
    scratch: Vec<GpuQuad>,
}

impl Renderer {
    /// Build the render pipeline and camera bind group. Shared by all targets so
    /// the GPU-resource code is type-checked on native builds even though the
    /// surface itself is only created on the web (see [`Renderer::new`]).
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    fn build(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> (
        wgpu::RenderPipeline,
        wgpu::Buffer,
        wgpu::BindGroup,
        wgpu::Buffer,
    ) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quad shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("quad.wgsl").into()),
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("quad pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Per-instance vertex layout: center, half_extent, color.
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuQuad>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("quad pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[instance_layout],
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

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quad instances"),
            size: (std::mem::size_of::<GpuQuad>() as u64).max(1),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        (pipeline, camera_buffer, camera_bind_group, instance_buffer)
    }

    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    fn from_parts(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
    ) -> Self {
        let (pipeline, camera_buffer, camera_bind_group, instance_buffer) =
            Self::build(&device, config.format);
        Self {
            device,
            queue,
            surface,
            config,
            pipeline,
            camera_buffer,
            camera_bind_group,
            instance_buffer,
            instance_capacity: 1,
            scratch: Vec::new(),
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

        // Gather all quad instances, in submission order. With only one
        // primitive kind, concatenation preserves render order; once other
        // kinds exist this becomes a per-batch loop with its own draw call.
        self.scratch.clear();
        for primitive in &frame.primitives {
            match primitive {
                Primitive::Quads(quads) => {
                    self.scratch.extend(quads.iter().map(GpuQuad::from));
                }
            }
        }
        let instance_count = self.scratch.len() as u32;
        if instance_count > self.instance_capacity {
            self.instance_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("quad instances"),
                        contents: bytemuck::cast_slice(&self.scratch),
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    });
            self.instance_capacity = instance_count;
        } else if instance_count > 0 {
            self.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.scratch),
            );
        }

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
                label: Some("quad pass"),
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

            if instance_count > 0 {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
                pass.draw(0..4, 0..instance_count);
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
