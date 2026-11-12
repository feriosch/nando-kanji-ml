// The runtimes step 6 of nando-kanji-ml#1 compares, pinned, each loaded from jsDelivr.
const CDN = "https://cdn.jsdelivr.net/npm";

export const LIBS = {
  tfjs: `${CDN}/@tensorflow/tfjs@4.22.0/dist/tf.min.js`,
  tfjsWasm: `${CDN}/@tensorflow/tfjs-backend-wasm@4.22.0/dist/tf-backend-wasm.min.js`,
  tfjsWasmDir: `${CDN}/@tensorflow/tfjs-backend-wasm@4.22.0/dist/`,
  ortWasm: `${CDN}/onnxruntime-web@1.31.1/dist/ort.wasm.min.js`,
  ortWebgpu: `${CDN}/onnxruntime-web@1.31.1/dist/ort.webgpu.min.js`,
  ortDir: `${CDN}/onnxruntime-web@1.31.1/dist/`,
  litert: `${CDN}/@litertjs/core@2.6.0/+esm`,
  litertWasmDir: `${CDN}/@litertjs/core@2.6.0/wasm/`,
};

export const VERSIONS = { tfjs: "TensorFlow.js 4.22.0", ort: "ONNX Runtime Web 1.31.1", litert: "LiteRT.js 2.6.0" };

const MODELS = {
  tfjs: { f32: "models/tfjs/model.json", f16: "models/tfjs_f16/model.json" },
  ort: { f32: "models/kanji.onnx", f16: "models/kanji_f16.onnx" },
  litert: { f32: "models/kanji.tflite", f16: "models/kanji_f16.tflite" },
};

const BACKENDS = { tfjs: ["webgl", "wasm"], ort: ["wasm", "webgpu"], litert: ["wasm", "webgpu"] };

export const VARIANTS = Object.entries(BACKENDS).flatMap(([runtime, backends]) =>
  backends.flatMap((backend) =>
    ["f32", "f16"].map((precision) => ({
      id: `${runtime}-${backend}-${precision}`,
      runtime,
      backend,
      precision,
      model: MODELS[runtime][precision],
    })),
  ),
);
