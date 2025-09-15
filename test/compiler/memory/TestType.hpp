#pragma once
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace denox::testing {

// Shared, external stats you pass into each Probe<T>.
struct OperationStats {
  // ctor counts
  std::size_t default_ctor = 0;
  std::size_t value_ctor   = 0; // in_place/value-init ctor (with args)
  std::size_t copy_ctor    = 0;
  std::size_t move_ctor    = 0;

  // assignment counts
  std::size_t copy_assign  = 0;
  std::size_t move_assign  = 0;

  // lifetime
  std::size_t dtor         = 0;
  std::size_t alive        = 0;
  std::size_t max_alive    = 0;

  void on_create() {
    ++alive;
    if (alive > max_alive) max_alive = alive;
  }
  void on_destroy() {
    ++dtor;
    if (alive == 0) throw std::runtime_error("OperationStats underflow");
    --alive;
  }

  std::size_t total_constructs() const {
    return default_ctor + value_ctor + copy_ctor + move_ctor;
  }
};

// RAII checker you can put at the end of a scope to ensure no leaks.
struct LeakSentinel {
  OperationStats& s;
  ~LeakSentinel() noexcept(false) {
    if (s.alive != 0) {
      throw std::runtime_error("LeakSentinel: objects still alive at scope end");
    }
    if (s.total_constructs() != s.dtor) {
      throw std::runtime_error("LeakSentinel: ctor/dtor count mismatch");
    }
  }
};

template<
  class T = int,
  bool Movable         = std::move_constructible<T>,
  bool Copyable        = std::copy_constructible<T>,
  bool MoveAssignable  = std::is_move_assignable_v<T>,
  bool CopyAssignable  = std::is_copy_assignable_v<T>
>
class Probe {
public:
  // Always require an external stats block.
  explicit Probe(OperationStats& stats)
    requires std::default_initializable<T>
  : stats_(&stats) {
    stats_->default_ctor++;
    stats_->on_create();
  }

  // Value/in-place constructor for T.
  template<class... Args>
  explicit Probe(OperationStats& stats, std::in_place_t, Args&&... args)
    requires std::constructible_from<T, Args...>
  : value_(std::forward<Args>(args)...), stats_(&stats) {
    stats_->value_ctor++;
    stats_->on_create();
  }

  // Copy ctor
  Probe(const Probe& other)
    requires (Copyable && std::copy_constructible<T>)
  : value_(other.checked().value_), stats_(other.stats_) {
    stats_->copy_ctor++;
    stats_->on_create();
  }

  // Move ctor
  Probe(Probe&& other) noexcept(std::is_nothrow_move_constructible_v<T>)
    requires (Movable && std::move_constructible<T>)
  : value_(std::move(other.checked().value_)), stats_(other.stats_) {
    stats_->move_ctor++;
    stats_->on_create();
    other.moved_from_ = true;
  }

  // Copy assign
  Probe& operator=(const Probe& other)
    requires (CopyAssignable && std::is_copy_assignable_v<T>)
  {
    if (this == &other) return *this;
    checked(); other.checked();
    value_ = other.value_;
    stats_ = other.stats_;
    stats_->copy_assign++;
    return *this;
  }

  // Move assign
  Probe& operator=(Probe&& other) noexcept(std::is_nothrow_move_assignable_v<T>)
    requires (MoveAssignable && std::is_move_assignable_v<T>)
  {
    if (this == &other) return *this;
    checked(); other.checked();
    value_  = std::move(other.value_);
    stats_  = other.stats_;
    stats_->move_assign++;
    other.moved_from_ = true;
    return *this;
  }

  // Deleted when disabled via template flags
  Probe(const Probe&) requires(!Copyable) = delete;
  Probe& operator=(const Probe&) requires(!CopyAssignable) = delete;
  Probe(Probe&&)      requires(!Movable) = delete;
  Probe& operator=(Probe&&) requires(!MoveAssignable) = delete;

  ~Probe() {
    if (stats_) stats_->on_destroy();
  }

  // Observers/Accessors – guarded against use-after-move
  T&       get()       { return checked().value_; }
  const T& get() const { return checked().value_; }

  T*       operator->()       { return &checked().value_; }
  const T* operator->() const { return &checked().value_; }

  T&       operator*()       { return checked().value_; }
  const T& operator*() const { return checked().value_; }

  bool moved_from() const noexcept { return moved_from_; }

  OperationStats& stats()       { ensure_stats_(); return *stats_; }
  const OperationStats& stats() const { ensure_stats_(); return *stats_; }

private:
  // Throws if used after move, or untracked.
  const Probe& checked() const {
    if (!stats_) throw std::runtime_error("Probe: no stats bound");
    if (moved_from_) throw std::runtime_error("Probe: use-after-move");
    return *this;
  }
  Probe& checked() { (void)static_cast<const Probe&>(*this).checked(); return *this; }

  void ensure_stats_() const {
    if (!stats_) throw std::runtime_error("Probe: no stats bound");
  }

  T value_{};
  OperationStats* stats_ = nullptr; // non-owning
  bool moved_from_ = false;
};

} // namespace denox::testing
