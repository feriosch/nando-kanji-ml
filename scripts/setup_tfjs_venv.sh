#!/usr/bin/env bash
# Creates .venv-tfjs, where `python -m kanji_ml.export` finds the TensorFlow.js converter.
#
# tensorflowjs 4.22.0 (the latest, from 2024) can't share the training venv: it
# pins TensorFlow 2.19. It also imports tensorflow_decision_forests on every
# conversion, but that package's only release for TF 2.19 needs protobuf >= 6.31,
# which TF 2.19 can't use. The converter only needs it for decision-forest
# models, so an empty stand-in module takes its place.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv-tfjs
.venv-tfjs/bin/pip install --quiet --upgrade pip
.venv-tfjs/bin/pip install --quiet tensorflowjs==4.22.0 tensorflow==2.19.0 tf_keras==2.19.0 \
  protobuf==5.29.6 "setuptools<81"  # setuptools 81 dropped pkg_resources, which the converter imports
.venv-tfjs/bin/pip uninstall --quiet --yes tensorflow-decision-forests ydf || true

site=$(.venv-tfjs/bin/python -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$site/tensorflow_decision_forests"
cat > "$site/tensorflow_decision_forests/__init__.py" <<'EOF'
# Stand-in: tensorflowjs 4.22 imports this unconditionally but only uses it for
# decision-forest models. The real package needs protobuf>=6.31, which TF 2.19 can't use.
EOF

.venv-tfjs/bin/tensorflowjs_converter --version
