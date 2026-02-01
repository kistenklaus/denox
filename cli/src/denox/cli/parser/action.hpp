#pragma once

#include "denox/cli/parser/actions/bench.hpp"
#include "denox/cli/parser/actions/compile.hpp"
#include "denox/cli/parser/actions/dumpcsv.hpp"
#include "denox/cli/parser/actions/help.hpp"
#include "denox/cli/parser/actions/infer.hpp"
#include "denox/cli/parser/actions/populate.hpp"

#include <cassert>
#include <variant>

enum class ActionKind {
  Compile,
  Infer,
  Bench,
  Populate,
  Help,
  Version,
  DumpCsv
};

class Action {
public:
  Action(CompileAction a) noexcept : m_value(std::move(a)) {}

  Action(InferAction a) noexcept : m_value(std::move(a)) {}

  Action(BenchAction a) noexcept : m_value(std::move(a)) {}

  Action(PopulateAction a) noexcept : m_value(std::move(a)) {}

  Action(HelpAction a) noexcept : m_value(std::move(a)) {}

  Action(DumpCsvAction a) noexcept : m_value(std::move(a)) {}

  static Action version() noexcept { return Action{VersionTag{}}; }

  ActionKind kind() const noexcept {
    if (std::holds_alternative<CompileAction>(m_value))
      return ActionKind::Compile;
    if (std::holds_alternative<InferAction>(m_value))
      return ActionKind::Infer;
    if (std::holds_alternative<BenchAction>(m_value))
      return ActionKind::Bench;
    if (std::holds_alternative<PopulateAction>(m_value))
      return ActionKind::Populate;
    if (std::holds_alternative<HelpAction>(m_value))
      return ActionKind::Help;
    if (std::holds_alternative<VersionTag>(m_value))
      return ActionKind::Version;
    if (std::holds_alternative<DumpCsvAction>(m_value)) {
      return ActionKind::DumpCsv;
    }
    std::abort();
  }

  const CompileAction &compile() const noexcept {
    assert(kind() == ActionKind::Compile);
    return std::get<CompileAction>(m_value);
  }

  CompileAction &compile() noexcept {
    assert(kind() == ActionKind::Compile);
    return std::get<CompileAction>(m_value);
  }

  const InferAction &infer() const noexcept {
    assert(kind() == ActionKind::Infer);
    return std::get<InferAction>(m_value);
  }

  InferAction &infer() noexcept {
    assert(kind() == ActionKind::Infer);
    return std::get<InferAction>(m_value);
  }

  const BenchAction &bench() const noexcept {
    assert(kind() == ActionKind::Bench);
    return std::get<BenchAction>(m_value);
  }

  BenchAction &bench() noexcept {
    assert(kind() == ActionKind::Bench);
    return std::get<BenchAction>(m_value);
  }

  const PopulateAction &populate() const noexcept {
    assert(kind() == ActionKind::Populate);
    return std::get<PopulateAction>(m_value);
  }

  PopulateAction &populate() noexcept {
    assert(kind() == ActionKind::Populate);
    return std::get<PopulateAction>(m_value);
  }

  const HelpAction &help() const noexcept {
    assert(kind() == ActionKind::Help);
    return std::get<HelpAction>(m_value);
  }

  const DumpCsvAction &dumpCsv() const noexcept {
    assert(kind() == ActionKind::DumpCsv);
    return std::get<DumpCsvAction>(m_value);
  }

  DumpCsvAction &dumpcsv() noexcept {
    assert(kind() == ActionKind::DumpCsv);
    return std::get<DumpCsvAction>(m_value);
  }

private:
  struct VersionTag {};

  explicit Action(VersionTag) noexcept : m_value(VersionTag{}) {}

private:
  std::variant<CompileAction, InferAction, BenchAction, PopulateAction,
               HelpAction, VersionTag, DumpCsvAction>
      m_value;
};
