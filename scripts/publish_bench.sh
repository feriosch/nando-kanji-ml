#!/usr/bin/env bash
# Publishes bench/ to the gh-pages branch, which GitHub Pages serves at
# https://feriosch.github.io/nando-kanji-ml/. Run `python -m kanji_ml.export` first:
# the page, its models and reference.json then all come from the same export.
#
#   scripts/publish_bench.sh            # commit and push
#   scripts/publish_bench.sh --dry-run  # commit in a temporary worktree, show it, push nothing
set -euo pipefail
cd "$(dirname "$0")/.."

[[ -f bench/reference.json ]] || { echo "no bench/reference.json: run python -m kanji_ml.export first" >&2; exit 1; }
missing=$(python3 - <<'EOF'
import json, pathlib
print(" ".join(p for p in json.load(open("bench/reference.json", encoding="utf-8"))["exported"]
               if not pathlib.Path("bench", p).exists()))
EOF
)
[[ -z "$missing" ]] || { echo "reference.json lists models that aren't in bench/: $missing" >&2; exit 1; }

source=$(git rev-parse --short HEAD)
[[ -z "$(git status --porcelain bench)" ]] || source="$source with uncommitted bench/ changes"
site=$(mktemp -d)
trap 'git worktree remove --force "$site" >/dev/null 2>&1 || true; rm -rf "$site"' EXIT

git fetch --quiet origin gh-pages
git worktree add --quiet --detach "$site" origin/gh-pages
find "$site" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp bench/index.html bench/run.html bench/run.js bench/variants.js bench/reference.json "$site/"
cp -r bench/models "$site/models"
touch "$site/.nojekyll"  # serve the files as they are, without GitHub's Jekyll build

git -C "$site" add --all
if git -C "$site" diff --cached --quiet; then
  echo "gh-pages already has this bench: nothing to publish"
  exit 0
fi
git -C "$site" commit --quiet -m "Publish the runtime bench from $source" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git -C "$site" show --stat --format="%h %s" HEAD | head -20

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "dry run: nothing pushed"
else
  git -C "$site" push --quiet origin HEAD:gh-pages
  echo "pushed: https://feriosch.github.io/nando-kanji-ml/ updates within a few minutes (pages are cached up to 10)"
fi
