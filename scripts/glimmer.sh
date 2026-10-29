#!/bin/sh
# Muse Glimmer (open weights, 30B) on this Mac for the offline answer path: scripts/glimmer.sh [port]
# Weights: hf download meta-models/Muse-Glimmer-30B-GGUF --local-dir ~/helloworld/models/muse-glimmer \
#   --include "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"   (17 GB; holds ~18 GB of memory while up)
# Then set providers.offline = "glimmer" in metasight.toml.
set -e
M="${GLIMMER_DIR:-$HOME/helloworld/models/muse-glimmer}/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
exec nice -n 19 taskpolicy -b llama-server -m "$M" --host 127.0.0.1 --port "${1:-8091}" \
  -c 8192 -ngl 99 --jinja -a muse-glimmer
