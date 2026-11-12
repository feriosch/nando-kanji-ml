# nando-kanji-ml

Trains the model that recognises a kanji the Owner draws in Nando (a **Drawing**), and publishes it to nando-web, where it runs in the browser. The design is in nando-server's ADR 0005; the plan is [#1](https://github.com/feriosch/nando-kanji-ml/issues/1).

It continues the HanCharPI college model, retrained on ETL9G's 2,965 kanji.

## Setup (Windows + WSL2)

Training runs in **WSL2 with Ubuntu 24.04** on the RTX 3070. TensorFlow has had no GPU support on native Windows since 2.10.

1. Install the NVIDIA driver on **Windows only**, never inside WSL. In Ubuntu, `nvidia-smi` should list the GPU.
2. Give WSL more memory than its default of half the RAM. In `C:\Users\<you>\.wslconfig`:
   ```ini
   [wsl2]
   memory=10GB
   swap=8GB
   ```
   Then run `wsl --shutdown` from PowerShell.
3. Create the environment, from the Linux home folder (not under `/mnt/c` or `/mnt/e`, which are slow):
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt -r requirements-ci.txt
   ```
4. **Let TensorFlow find cusolver.** TensorFlow 2.21 finds every pip-installed CUDA library except `libcusolver.so.11`, and then quietly falls back to the CPU. Append this to `.venv/bin/activate`, then run `deactivate && source .venv/bin/activate`:
   ```bash
   # Let TensorFlow find the pip-installed CUDA libraries
   NVIDIA_LIBS=$(python -c "import nvidia, os; print(':'.join(os.path.join(p, d, 'lib') for p in nvidia.__path__ for d in os.listdir(p) if os.path.isdir(os.path.join(p, d, 'lib'))))")
   export LD_LIBRARY_PATH="$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
   ```
5. Check that this lists `GPU:0`:
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

## Data

ETL9G (from the [ETL Character Database](https://etlcdb.db.aist.go.jp/)) is for non-commercial use and must not be redistributed, so it never goes into git. The official site currently times out. A private backup is at `gs://nanders-backups/datasets/ETL9G.zip`.

Unzip it so that `data/ETL9G/` holds `ETL9G_01` to `ETL9G_50`. Each file holds 4 writers × 3,036 characters (2,965 kanji and 71 hiragana).

### Build the training files

```bash
python -m kanji_ml.dataset
```

This takes about 3 minutes on 8 cores and writes three files to `data/prepared/`:
- `labels.json`: the 2,965 kanji in JIS order. A model output `i` means `labels[i]`.
- `train.npz`: files 01–45 (180 writers), 533,700 samples.
- `test.npz`: files 46–50 (20 writers), 59,300 samples.

Hiragana are dropped. The split is by writer, so the test score measures handwriting the model has never seen.

First, `etl9g.without_form_marks` removes the pieces of the paper form that the scan caught around each sample: the borders and guide lines of the neighbouring boxes, and the ends of strokes written in them. Without this, about 3 in 4 samples would be cropped too wide. The step is for ETL images only, because a Drawing has no form around it.

Each sample is then preprocessed by `kanji_ml/preprocess.py`, which the app will copy exactly in TypeScript:
1. Pixels at or below a fifth of the strongest one become background.
2. The image is cropped to the ink.
3. It's scaled by area-averaging so its longer side is 40 pixels.
4. It's centred in 48×48.
5. It's stretched so the strongest ink is 255, as ink 255 on a 0 background.

### Drawings

The app builds the model's input from a Drawing's strokes, not from the screen's pixels. `kanji_ml/strokes.py` defines how, and the app copies it step for step in TypeScript. The spec and its reasons are on [#1](https://github.com/feriosch/nando-kanji-ml/issues/1).
1. The strokes are scaled to fit a 160-pixel box, 4× the model's 40 pixels. Where and how big the Owner drew doesn't matter. Strokes with no points are skipped.
2. They're drawn with a soft pen on a 164-pixel square, measured at each pixel's centre. Ink fades from 1 on a stroke's centre line to 0 at 6 pixels away, like ETL's scanned strokes. Where strokes overlap, the stronger ink wins.
3. The ink is stored as whole numbers, `floor(ink × 255 + 0.5)`.
4. The image goes through `preprocess.py` as above.

`fixtures/drawings.json` holds Drawings and the 48×48 input each must give, and each release copies it to nando-web. Python must match it exactly. The TypeScript copy may differ by 1 per pixel, because `preprocess.py`'s area-averaging is a matrix product, and NumPy adds it up in a different order from a plain loop. Regenerate the file only when the preprocessing changes on purpose:

```bash
python -m kanji_ml.fixtures
```

## Train

```bash
python -m kanji_ml.train
```

It trains on `data/prepared/` and writes `models/kanji.keras` (the best epoch), `history.csv` and `metrics.json`. On the RTX 3070 it takes about 30 minutes.
- **Model** (`kanji_ml/model.py`): three conv blocks (64, 128 and 256 filters, each two 3×3 convolutions with batch normalisation), then global average pooling, dropout and a softmax over the 2,965 kanji. That's 1,908,949 parameters, about 7.6 MB as float32. It takes the preprocessed image as it is, 0–255.
- **Validation:** the last 2 training files (8 writers) are held out for early stopping, so validation is also on unseen writers. The test set is used once, at the end. `dataset.hold_out_last_files` checks that the samples are whole files in file order, so writers can't leak across.
- **Augmentation** (`kanji_ml/augment.py`): rotation ±10°, zoom ±10% and shifts of ±6%, plus mild stroke-width variation. A third of the images get about a pixel thicker and a third about a pixel thinner. Drawings already match ETL's typical stroke (see the spec on #1), so this only keeps the model from relying on one exact width.
- **Training:** Adam at 1e-3, mixed precision, batches of 256, with every sample shuffled each epoch. The learning rate drops ×0.3 when validation loss stalls for 2 epochs, and training stops after 5.

The script ends by printing top-1 and top-8 on the test set (20 writers the model has never seen), and whether top-8 passes the 95% shipping gate (ADR 0005).

### Results

| Version | Top-1 | Top-8 | Test samples | Epochs | Notes |
|---|---|---|---|---|---|
| v1 | 99.42% | 99.92% | 59,300 | 24 (best: 19) | With stroke-width augmentation. The first run, without it, scored 99.40% / 99.90%. |

## Export and bench

The app runs the model with one of three browser runtimes: TensorFlow.js, LiteRT.js or ONNX Runtime Web. The bench page in `bench/` decides which one by measuring each on the iPhone 17.

One-time setup, besides the training venv:

```bash
pip install -r requirements-export.txt
scripts/setup_tfjs_venv.sh
```

`scripts/setup_tfjs_venv.sh` creates `.venv-tfjs` for the TensorFlow.js converter. tensorflowjs 4.22, the latest, pins an older TensorFlow and needs a workaround that the script explains.

Then:

```bash
python -m kanji_ml.export
```

This writes `bench/models/` and `bench/reference.json`, which aren't in git:
- `tfjs/`, `kanji.tflite` and `kanji.onnx`, plus a float16 copy of each, which is half the size.
- Keras's top 8 for every fixture Drawing.

It rebuilds the model in float32 first, because training's mixed precision leaves float16 operations that LiteRT can't run. It also uses a fixed batch of 1. It checks the LiteRT and ONNX exports against Keras. TensorFlow.js only runs in a browser, so the bench page checks that one.

To try the bench locally, serve `bench/` (for example `python -m http.server -d bench`) and open it. For each runtime and backend, in float32 and float16, it shows:
- the download size (runtime plus model);
- the load time, the first run, and the median run time;
- whether the top 1 and top 8 match Keras.

The published bench is at https://feriosch.github.io/nando-kanji-ml/, served from the `gh-pages` branch. To update it, re-run the export, then run `scripts/publish_bench.sh`. It publishes the page, the models and `reference.json` from that one export, refuses if a model is missing, and records the source commit. Add `--dry-run` to see the commit without pushing.

### Runtime choice

**The app uses LiteRT.js 2.6.0 on its WASM (CPU) backend, with the float16 model.** The Owner chose it on 2026-11-12 after measuring on their iPhone 17 (Safari 26.6).

v1, iPhone 17, with each runtime recognising the 19 fixture Drawings one at a time:

| Runtime · backend · precision | Download | Per run (median of 50) | 1st run | Top-1 = Keras | Top-8 = Keras |
|---|---|---|---|---|---|
| **LiteRT.js · WASM · float16** | **6.1 MB** | **16 ms** | 32 ms | 19/19 | **19/19** |
| LiteRT.js · WASM · float32 | 9.6 MB | 16 ms | 23 ms | 19/19 | 19/19 |
| LiteRT.js · WebGPU · float16 | 6.1 MB | 2.0 ms | 12 ms | 19/19 | 15/19 |
| LiteRT.js · WebGPU · float32 | 9.6 MB | 3.0 ms | 21 ms | 19/19 | 16/19 |
| TensorFlow.js · WASM · float16 | 4.0 MB | 12 ms | 25 ms | 19/19 | 19/19 |
| TensorFlow.js · WASM · float32 | 7.6 MB | 11 ms | 49 ms | 19/19 | 19/19 |
| TensorFlow.js · WebGL · float16 | 3.9 MB | 1000 ms | 192 ms | 19/19 | 19/19 |
| TensorFlow.js · WebGL · float32 | 7.4 MB | 1000 ms | 107 ms | 19/19 | 19/19 |
| ONNX Runtime Web · WASM · float16 | 6.7 MB | 14 ms | 22 ms | 19/19 | 19/19 |
| ONNX Runtime Web · WASM · float32 | 10.3 MB | 13 ms | 24 ms | 19/19 | 19/19 |
| ONNX Runtime Web · WebGPU · float16 | 9.5 MB | 4.0 ms | 19 ms | 19/19 | 15/19 |
| ONNX Runtime Web · WebGPU · float32 | 13.1 MB | 4.0 ms | 20 ms | 19/19 | 19/19 |

Versions: LiteRT.js 2.6.0, TensorFlow.js 4.22.0, ONNX Runtime Web 1.31.1. Download is the runtime plus the model, as compressed bytes over the network.

**Why LiteRT.js on WASM, in float16:**
- **It's maintained.** LiteRT.js is Google's current browser runtime (2.6.0, October 2026). TensorFlow.js hasn't had a release since October 2024. The app will live for years, and a runtime that stops keeping up with Safari would mean a forced switch later.
- **It gives up very little.** TensorFlow.js on WASM was smallest and fastest (4.0 MB, 12 ms), which is what #1's rule asks for. LiteRT.js costs 2.1 MB more, downloaded once and then cached by the service worker (ADR 0005), and 4 ms more per Drawing. Neither is noticeable.
- **It matches Keras exactly** on all 19 fixtures, top 8 included. ONNX Runtime on WASM also does, but it's bigger (6.7 MB) and no faster.
- **Float16 halves the model to 3.8 MB** and still matches Keras exactly. Float32 is no faster.
- **WebGPU is faster (2–4 ms), but the difference can't be felt.** It also reorders Keras's low-probability candidates (ranks 4–8). The same runtime can switch to it later if needed, since it's one option to `loadAndCompile`.

**For the app (nando-web#50):**
- Load it with `loadLiteRt(<wasm dir>)`, then `loadAndCompile(<model>, {accelerator: 'wasm'})`. `bench/run.js` shows a working version.
- LiteRT.js loads its WASM files from the directory passed to `loadLiteRt`. For Draw to work offline, serve that directory with the app, or make sure the service worker caches it.

**Caveats:**
- **Download sizes were measured on the published bench in desktop Chrome,** because Safari reports cached files as 0 bytes. The bytes are the same on any device, except possibly for WebGPU, where Safari may fetch different runtime files.
- **The iPhone's cold load times aren't in the table.** Most runs reused files from an earlier run.
- **TensorFlow.js on WebGL took about 1 second per run.** This is likely an artifact of the bench, but it isn't verified. TensorFlow.js waits for the GPU with a timer, and Safari slows timers to once a second in a frame that's out of view, as the bench's frame was then. The frame is now pinned on screen, but the iPhone hasn't re-run the bench since.

## Release

The app runs the model in `release/`, which is in git:
- `kanji.tflite`: the float16 LiteRT model, for LiteRT.js on WASM (see "Runtime choice");
- `labels.json`: output `i` is the kanji `labels[i]`;
- `release.json`: the runtime, accelerator, input shape and test accuracy.

Rebuild it after training:

```bash
python -m kanji_ml.release
```

It refuses to write anything unless the model matches Keras's top 8 on every fixture Drawing, the labels match its outputs, and `models/metrics.json` gives the test accuracy to record.

To ship a version, merge `release/` into `main` and tag a commit in `main`:

```bash
git tag v1 && git push origin v1
```

The `Release` workflow then opens a PR in nando-web, replacing `public/models/kanji/` with:
- the release files;
- `fixtures/drawings.json`;
- a `version.json` holding the tag.

It needs the secret `NANDO_WEB_TOKEN`: a fine-grained token limited to feriosch/nando-web, with *Contents* and *Pull requests* set to read and write. If nando-web already has a `kanji-model/<tag>` branch, the workflow stops rather than overwrite it. To release a tag again, close that PR and delete the branch first.

On this repo's PRs that touch the release, the workflow runs a dry run without the token. It does everything except the push and opening the PR.

Each version's accuracy is in "Results" above, and `release/release.json` records exactly what shipped.

## Tests

```bash
pytest
```

The tests use small fake ETL9G files, so they run without the dataset. CI installs only `requirements-ci.txt`, so the model tests, which need TensorFlow, run only on the training machine.
