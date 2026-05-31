// Instanced solid-colored quads.
//
// One draw call renders N quads as a 4-vertex triangle strip per instance. The
// unit corner is derived from the vertex index; per-instance center/half_extent/
// color come from the instance buffer; the camera uniform maps world -> clip.

struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

struct Instance {
    @location(0) center: vec2<f32>,
    @location(1) half_extent: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, inst: Instance) -> VertexOut {
    // vertex_index 0..4 -> unit corners (-1,-1), (1,-1), (-1,1), (1,1).
    let unit = vec2<f32>(f32(vertex_index & 1u), f32(vertex_index >> 1u)) * 2.0 - 1.0;
    let world = inst.center + unit * inst.half_extent;

    var out: VertexOut;
    out.clip_position = camera.view_proj * vec4<f32>(world, 0.0, 1.0);
    out.color = inst.color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}
