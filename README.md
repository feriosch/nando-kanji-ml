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

The app builds the model's input from a Drawing's strokes, not from the screen's pixels. `kanji_ml/strokes.py` defines how, and the app copies it exactly in TypeScript. The spec and its reasons are on [#1](https://github.com/feriosch/nando-kanji-ml/issues/1).
1. The strokes are scaled to fit a 160-pixel box, 4× the model's 40 pixels. Where and how big the Owner drew doesn't matter.
2. They're drawn with a soft pen: ink fades from the centre line to nothing 6 pixels away, like ETL's scanned strokes. Where strokes overlap, the stronger ink wins.
3. The image goes through `preprocess.py` as above.

`fixtures/drawings.json` holds Drawings and the 48×48 input each must give. Both repos test against it, allowing a difference of 1 per pixel, and each release copies it to nando-web. Regenerate it only when the preprocessing changes on purpose:

```bash
python -m kanji_ml.fixtures
```

## Tests

```bash
pytest
```

The tests use small fake ETL9G files, so they run without the dataset. CI installs only `requirements-ci.txt`.
