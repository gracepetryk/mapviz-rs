// Instanced textured quads (e.g. map tiles).
//
// One draw call renders a 4-vertex triangle strip per instance. The unit corner
// comes from the vertex index; per-instance center/half_extent come from the
// instance buffer; the camera uniform maps world -> clip. The bound texture is
// sampled across the quad. UV.v is flipped so the image's top row (v=0) lands at
// the top of the quad in world space (where +y is up).

struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(1) @binding(0)
var tex: texture_2d<f32>;
@group(1) @binding(1)
var tex_sampler: sampler;

struct Instance {
    @location(0) center: vec2<f32>,
    @location(1) half_extent: vec2<f32>,
};

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, inst: Instance) -> VertexOut {
    // vertex_index 0..4 -> unit corners (-1,-1), (1,-1), (-1,1), (1,1).
    let corner = vec2<f32>(f32(vertex_index & 1u), f32(vertex_index >> 1u));
    let unit = corner * 2.0 - 1.0;
    let world = inst.center + unit * inst.half_extent;

    var out: VertexOut;
    out.clip_position = camera.view_proj * vec4<f32>(world, 0.0, 1.0);
    // corner.x in [0,1] -> u; flip v so image top maps to world top.
    out.uv = vec2<f32>(corner.x, 1.0 - corner.y);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(tex, tex_sampler, in.uv);
}
