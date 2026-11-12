// Runs one variant (?v=<id>) in this page alone, so its download size and load time are its own.
// Reports to the parent page with postMessage, and shows the result here too.
import { LIBS, VARIANTS } from "./variants.js";

const SHAPE = [1, 48, 48, 1];
const WARMUP = 5;
const TIMED = 50;

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.crossOrigin = "anonymous";
    script.onload = resolve;
    script.onerror = () => reject(new Error(`couldn't load ${src}`));
    document.head.append(script);
  });
}

// Each runtime: load it and the model, and return predict(Float32Array) -> Promise<Float32Array of scores>.
const RUNTIMES = {
  async tfjs({ backend, model }) {
    await loadScript(LIBS.tfjs);
    if (backend === "wasm") {
      await loadScript(LIBS.tfjsWasm);
      tf.wasm.setWasmPaths(LIBS.tfjsWasmDir);
    }
    if (!(await tf.setBackend(backend))) throw new Error(`TensorFlow.js has no ${backend} backend here`);
    await tf.ready();
    const graph = await tf.loadGraphModel(model);
    return async (pixels) => {
      const input = tf.tensor4d(pixels, SHAPE);
      const output = graph.predict(input);
      const scores = await output.data();
      tf.dispose([input, output]);
      return scores;
    };
  },

  async ort({ backend, model }) {
    if (backend === "webgpu" && !navigator.gpu) throw new Error("WebGPU isn't available in this browser");
    await loadScript(backend === "webgpu" ? LIBS.ortWebgpu : LIBS.ortWasm);
    ort.env.wasm.wasmPaths = LIBS.ortDir;
    ort.env.wasm.numThreads = 1; // threads need cross-origin isolation, which a plain static host doesn't give
    const session = await ort.InferenceSession.create(model, { executionProviders: [backend] });
    const [inputName] = session.inputNames;
    const [outputName] = session.outputNames;
    return async (pixels) => {
      const outputs = await session.run({ [inputName]: new ort.Tensor("float32", pixels, SHAPE) });
      return outputs[outputName].data;
    };
  },

  async litert({ backend, model }) {
    if (backend === "webgpu" && !navigator.gpu) throw new Error("WebGPU isn't available in this browser");
    const litert = await import(LIBS.litert);
    await litert.loadLiteRt(LIBS.litertWasmDir);
    const compiled = await litert.loadAndCompile(model, { accelerator: backend });
    return async (pixels) => {
      let input = new litert.Tensor(pixels, SHAPE);
      if (backend === "webgpu") {
        const onGpu = await input.moveTo("webgpu");
        if (onGpu !== input) input.delete?.();
        input = onGpu;
      }
      const outputs = await compiled.run(input);
      input.delete();
      const onCpu = await outputs[0].moveTo("wasm");
      const scores = onCpu.toTypedArray().slice(); // copy before the tensor's memory is freed
      onCpu.delete();
      for (const output of outputs) if (output !== onCpu && !output.isDeleted?.()) output.delete?.();
      return scores;
    };
  },
};

function topK(scores, k = 8) {
  return Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a] || a - b).slice(0, k);
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

// Bytes over the network for the runtime and model. A file served from the cache has transferSize 0.
// Chrome still reports its size; Safari reports 0, so those files are counted as unmeasured.
function downloaded() {
  let bytes = 0;
  let cached = 0;
  let unmeasured = 0;
  for (const entry of performance.getEntriesByType("resource")) {
    if (entry.name.includes("reference.json")) continue; // the bench's own test data, not the runtime's
    bytes += entry.encodedBodySize;
    if (entry.transferSize === 0) cached += 1;
    if (entry.transferSize === 0 && entry.encodedBodySize === 0) unmeasured += 1;
  }
  return { bytes, cached, unmeasured };
}

async function run(variant) {
  const reference = await (await fetch("reference.json")).json();
  if (!reference.exported.includes(variant.model)) throw new Error("not exported: run python -m kanji_ml.export with the TF.js venv");
  const inputs = reference.fixtures.map((f) => Float32Array.from(f.input.flat()));
  const loadStart = performance.now(); // after the page and test data, so this times only runtime + model
  const predict = await RUNTIMES[variant.runtime](variant);
  const loadMs = performance.now() - loadStart;

  let start = performance.now();
  const first = await predict(inputs[0]);
  const firstMs = performance.now() - start;

  let sameTop8 = 0;
  let sameTop1 = 0;
  const differences = []; // where the order differs from Keras: the first rank, and whether the 8 are the same set
  for (const [i, fixture] of reference.fixtures.entries()) {
    const top = topK(i === 0 ? first : await predict(inputs[i]));
    const rank = top.findIndex((label, r) => label !== fixture.keras_top8[r]);
    if (rank === -1) sameTop8 += 1;
    else differences.push({ fixture: fixture.name, firstDifferentRank: rank + 1,
                            sameSet: top.every((label) => fixture.keras_top8.includes(label)) });
    if (top[0] === fixture.keras_top8[0]) sameTop1 += 1;
  }

  for (let i = 0; i < WARMUP; i += 1) await predict(inputs[i % inputs.length]);
  const times = [];
  for (let i = 0; i < TIMED; i += 1) {
    start = performance.now();
    await predict(inputs[i % inputs.length]);
    times.push(performance.now() - start);
  }

  return {
    loadMs,
    firstMs,
    medianMs: median(times),
    sameTop8: `${sameTop8}/${reference.fixtures.length}`,
    sameTop1: `${sameTop1}/${reference.fixtures.length}`,
    differences,
    ...downloaded(),
  };
}

const id = new URLSearchParams(location.search).get("v");
const variant = VARIANTS.find((v) => v.id === id);
const out = document.querySelector("#out");
try {
  if (!variant) throw new Error(`no variant "${id}"`);
  out.textContent = `running ${id}…`;
  const result = await run(variant);
  out.textContent = JSON.stringify({ id, ...result }, null, 2);
  parent.postMessage({ id, result }, "*");
} catch (error) {
  out.textContent = `${id}: ${error.message ?? error}`;
  parent.postMessage({ id, error: String(error.message ?? error) }, "*");
}
