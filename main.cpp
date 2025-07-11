
#include "torch/torch.h"
#include <cstdio>

int main() {
  bool x = ::torch::cuda::is_available();
  std::printf("%d\n", x);
}
