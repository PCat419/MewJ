#!/usr/bin/env bash
# 在 Linux 上从 mahjong-cpp 编译 nanikiru，并安装到 MewJ-deploy/engine/
#
# 用法（在 MewJ-deploy 目录，或任意目录指定路径）：
#   bash deploy/build_nanikiru.sh /path/to/mahjong-cpp
#   MAHJONG_CPP=/path/to/mahjong-cpp MEWJ_ROOT=/opt/mewj bash deploy/build_nanikiru.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEWJ_ROOT="${MEWJ_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MAHJONG_CPP="${1:-${MAHJONG_CPP:-}}"

if [[ -z "$MAHJONG_CPP" ]]; then
  # 常见布局：Code/mahjong-cpp 与 Code/MewJ-deploy 并列
  for cand in \
    "$(cd "$MEWJ_ROOT/.." && pwd)/mahjong-cpp" \
    "$MEWJ_ROOT/../mahjong-cpp" \
    /opt/mahjong-cpp
  do
    if [[ -f "$cand/CMakeLists.txt" ]]; then
      MAHJONG_CPP="$(cd "$cand" && pwd)"
      break
    fi
  done
fi

if [[ -z "$MAHJONG_CPP" || ! -f "$MAHJONG_CPP/CMakeLists.txt" ]]; then
  echo "找不到 mahjong-cpp 源码目录。"
  echo "用法: bash deploy/build_nanikiru.sh /path/to/mahjong-cpp"
  exit 1
fi

MAHJONG_CPP="$(cd "$MAHJONG_CPP" && pwd)"
ENGINE="$MEWJ_ROOT/engine"
BUILD_DIR="$MAHJONG_CPP/build-linux"

echo "==> mahjong-cpp: $MAHJONG_CPP"
echo "==> MewJ engine: $ENGINE"
echo "==> build dir:   $BUILD_DIR"

need_pkgs=()
command -v cmake >/dev/null || need_pkgs+=(cmake)
command -v g++ >/dev/null || need_pkgs+=(build-essential)
command -v ninja >/dev/null || true
if ! dpkg -s libboost-filesystem-dev >/dev/null 2>&1; then
  need_pkgs+=(libboost-all-dev)
fi
if ! dpkg -s libomp-dev >/dev/null 2>&1; then
  need_pkgs+=(libomp-dev)
fi

if ((${#need_pkgs[@]})); then
  if [[ $EUID -eq 0 ]]; then
    apt-get update
    apt-get install -y "${need_pkgs[@]}"
  else
    echo "缺少编译依赖: ${need_pkgs[*]}"
    echo "请执行: sudo apt-get install -y ${need_pkgs[*]}"
    exit 1
  fi
fi

# OpenMP=ON 走 Linux 正常链接分支（默认 OFF 会误链 ws2_32）
# 只编 server，加快编译
cmake -S "$MAHJONG_CPP" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPENMP=ON \
  -DBUILD_SERVER=ON \
  -DBUILD_TEST=OFF \
  -DBUILD_SAMPLES=OFF \
  -DBUILD_TOOLS=OFF

cmake --build "$BUILD_DIR" -j"$(nproc)"

EXE="$BUILD_DIR/src/server/nanikiru"
if [[ ! -x "$EXE" ]]; then
  # install 布局兜底
  EXE="$BUILD_DIR/install/bin/nanikiru"
fi
if [[ ! -f "$EXE" ]]; then
  echo "编译后未找到 nanikiru: $BUILD_DIR/src/server/nanikiru"
  exit 1
fi

mkdir -p "$ENGINE"
install -m 755 "$EXE" "$ENGINE/nanikiru"

# 同步 schema（若源码侧更新了）
for f in request_schema.json response_schema.json; do
  if [[ -f "$MAHJONG_CPP/data/config/$f" ]]; then
    install -m 644 "$MAHJONG_CPP/data/config/$f" "$ENGINE/$f"
  fi
done

echo "已安装: $ENGINE/nanikiru"
"$ENGINE/nanikiru" --help >/dev/null 2>&1 || true
file "$ENGINE/nanikiru" || true
