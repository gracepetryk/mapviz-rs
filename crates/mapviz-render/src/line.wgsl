// Instanced solid-colored line segments.
//
// Each segment is expanded into a 4-vertex triangle strip: the endpoints offset
// by ±half-width along the segment's normal. Width is in world units, so it
// scales with zoom. Per-instance start/end/width/color come from the instance
// buffer; the camera uniform maps world -> clip.

struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

struct Instance {
    @location(0) start: vec2<f32>,
    @location(1) end: vec2<f32>,
    @location(2) width: f32,
    @location(3) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, inst: Instance) -> VertexOut {
    let dir = inst.end - inst.start;
    let len = length(dir);
    // Perpendicular to the segment; zero for a degenerate (zero-length) segment.
    var normal = vec2<f32>(0.0, 0.0);
    if (len > 1e-6) {
        let d = dir / len;
        normal = vec2<f32>(-d.y, d.x);
    }

    // vertex_index bit 1 selects the endpoint, bit 0 selects the side.
    let base = select(inst.start, inst.end, (vertex_index & 2u) != 0u);
    let side = select(-1.0, 1.0, (vertex_index & 1u) != 0u);
    let world = base + normal * (side * inst.width * 0.5);

    var out: VertexOut;
    out.clip_position = camera.view_proj * vec4<f32>(world, 0.0, 1.0);
    out.color = inst.color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}
