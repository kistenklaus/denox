#!/usr/bin/env sh

mkdir -p dist

docker run --rm -t \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  -v "$PWD":/work:ro \
  -v "$PWD/dist":/out \
  -w /tmp \
  ubuntu:22.04 \
  bash -c '
    set -euo pipefail

    apt update
    apt install -y \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        git \
        ca-certificates \
        libsqlite3-dev \
        zlib1g-dev \
        libvulkan-dev

    VULKAN_HEADERS_TAG="${VULKAN_HEADERS_TAG:-v1.4.341}"

    git clone --depth 1 --branch "$VULKAN_HEADERS_TAG" \
      https://github.com/KhronosGroup/Vulkan-Headers.git \
      /tmp/Vulkan-Headers

    cmake -S /tmp/Vulkan-Headers -B /tmp/Vulkan-Headers/build
    cmake --install /tmp/Vulkan-Headers/build --prefix /usr/local

    cmake -S /work -B /tmp/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DDENOX_STATIC_LIBSTDCPP=ON \
      -DDENOX_ENABLE_NVML_CLOCKCTRL=OFF \
      -DDENOX_SAN=OFF \
      -DBUILD_TESTING=OFF

    cmake --build /tmp/build -j"$(nproc)" --target denox
    cmake --install /tmp/build --prefix /tmp/install

    mkdir -p /tmp/package/denox-linux-x86_64
    cp -a /tmp/install/. /tmp/package/denox-linux-x86_64/

    tar -czf /out/denox-linux-x86_64.tar.gz -C /tmp/package denox-linux-x86_64
    chown "$HOST_UID:$HOST_GID" /out/denox-linux-x86_64.tar.gz
  '
