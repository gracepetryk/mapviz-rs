// Instanced solid-colored filled circles (discs).
//
// Each circle is drawn as a 4-vertex triangle-strip bounding quad of side
// 2*radius centered at `center`. The fragment shader discards anything outside
// the unit disc and smooths the edge with fwidth/smoothstep for sub-pixel
// anti-aliasing. Per-instance center/radius/color come from the instance
// buffer; the camera uniform maps world -> clip.

struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

struct Instance {
    @location(0) center: vec2<f32>,
    @location(1) radius: f32,
    @location(2) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    /// Local coordinates within the bounding quad, in [-1, 1] x [-1, 1].
    @location(0) local: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, inst: Instance) -> VertexOut {
    // vertex_index 0..4 -> unit corners (-1,-1), (1,-1), (-1,1), (1,1).
    let unit = vec2<f32>(f32(vertex_index & 1u), f32(vertex_index >> 1u)) * 2.0 - 1.0;
    let world = inst.center + unit * inst.radius;

    var out: VertexOut;
    out.clip_position = camera.view_proj * vec4<f32>(world, 0.0, 1.0);
    out.local = unit;
    out.color = inst.color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let dist = length(in.local);
    // Smooth the disc edge over one pixel's worth of local-space distance.
    let edge = fwidth(dist);
    let alpha = smoothstep(1.0, 1.0 - edge, dist);
    if (alpha <= 0.0) {
        discard;
    }
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
