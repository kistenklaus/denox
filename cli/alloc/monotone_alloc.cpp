#include "alloc/monotone_alloc.hpp"
#include <vector>

static std::vector<void *> s_allocations;

void *mono_alloc(size_t n) {
  void *x = malloc(n);
  s_allocations.push_back(x);
  return x;
}

void mono_free_all() {
  for (void *ptr : s_allocations) {
    free(ptr);
  }
}

