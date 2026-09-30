#!/usr/bin/env bash
set -euo pipefail
readonly BASELINE=58845ed63eb19aff55e896ea1f5d51f2a0df5b66
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/.tools/vcpkg}"
if [[ -e "$DEST" ]]; then
  if [[ ! -d "$DEST/.git" ]] || [[ "$(git -C "$DEST" rev-parse HEAD)" != "$BASELINE" ]]; then
    echo "Refusing to modify an existing directory/repository at $DEST. Use a new directory." >&2
    exit 2
  fi
else
  git clone https://github.com/microsoft/vcpkg.git "$DEST"
  git -C "$DEST" checkout --detach "$BASELINE"
fi
bash "$DEST/bootstrap-vcpkg.sh" -disableMetrics
printf '\nSet the toolchain root in your shell:\nexport VCPKG_ROOT=%q\n' "$DEST"
