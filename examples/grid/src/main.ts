import init, { Map } from "mapviz-wasm";
import wasmUrl from "mapviz-wasm/mapviz_bg.wasm?url";

const canvas = document.getElementById("map") as HTMLCanvasElement;
const errorEl = document.getElementById("error") as HTMLDivElement;

function showError(message: string): void {
  errorEl.textContent = message;
  errorEl.style.display = "grid";
}

/** Size the canvas backing store to its CSS size in physical pixels. */
function sizeCanvas(): [number, number] {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  canvas.width = width;
  canvas.height = height;
  return [width, height];
}

async function main(): Promise<void> {
  if (!("gpu" in navigator)) {
    showError(
      "WebGPU is not available in this browser.\nTry Chrome/Edge, or Safari 26+.",
    );
    return;
  }

  await init({ module_or_path: wasmUrl });

  sizeCanvas();

  let map: Map;
  try {
    map = await Map.create(canvas);
  } catch (e) {
    showError("Failed to initialize mapviz:\n" + String(e));
    return;
  }

  const dpr = () => window.devicePixelRatio || 1;

  // Keep the drawing surface in sync with the element's size.
  const observer = new ResizeObserver(() => {
    const [w, h] = sizeCanvas();
    map.resize(w, h);
  });
  observer.observe(canvas);

  // Pan with pointer drag.
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const d = dpr();
    map.pan((e.clientX - lastX) * d, (e.clientY - lastY) * d);
    lastX = e.clientX;
    lastY = e.clientY;
  });
  const endDrag = (e: PointerEvent) => {
    dragging = false;
    try {
      canvas.releasePointerCapture(e.pointerId);
    } catch {
      /* pointer may already be released */
    }
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  // Zoom toward the cursor with the wheel.
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const d = dpr();
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * d;
      const y = (e.clientY - rect.top) * d;
      // Scroll up (negative deltaY) zooms in; exponential keeps it smooth.
      map.zoom_at(Math.exp(-e.deltaY * 0.0015), x, y);
    },
    { passive: false },
  );

  // Render loop.
  const frame = () => {
    try {
      map.render();
    } catch (e) {
      showError("Render error:\n" + String(e));
      return; // stop the loop
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main();
