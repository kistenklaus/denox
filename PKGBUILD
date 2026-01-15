pkgname=denox
pkgver=0.1.0
pkgrel=1
pkgdesc="Denoise exporter for optimizing U-Nets for real-time rendering inference"
arch=('x86_64')
url="https://github.com/kistenklaus/denox"
license=('MIT')

depends=(
  'protobuf'
  'vulkan-icd-loader'
  'spirv-tools'
  'glslang'
  'libpng'
)

optdepends=(
  'fmt: use system fmt instead of bundled'
  'spdlog: use system spdlog instead of bundled'
  'vulkan-memory-allocator: use system VMA instead of bundled'
)

makedepends=(
  'cmake'
  'ninja'
  'flatbuffers'
  'protobuf'
)

source=("denox-$pkgver.tar.gz")
sha256sums=('SKIP')

build() {
  cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr

  cmake --build build
}

package() {
  DESTDIR="$pkgdir" cmake --install build
  install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
