#pragma once

#include "denox/cli/parser/actions/bench.hpp"
#include "denox/cli/parser/actions/compile.hpp"
#include "denox/cli/parser/actions/dumpcsv.hpp"
#include "denox/cli/parser/actions/help.hpp"
#include "denox/cli/parser/actions/infer.hpp"
#include "denox/cli/parser/actions/merge_device_info.hpp"
#include "denox/cli/parser/actions/populate.hpp"
#include "denox/cli/parser/actions/query_device_info.hpp"
#include "denox/cli/parser/actions/reweight.hpp"
#include "denox/diag/unreachable.hpp"

#include <cassert>
#include <variant>

enum class ActionKind {
  Compile,
  Infer,
  Bench,
  Populate,
  Help,
  Version,
  DumpCsv,
  Reweight,
  QueryDeviceInfo,
  MergeDeviceInfo,
};

class Action {
public:
  Action(CompileAction a) noexcept : m_value(std::move(a)) {}

  Action(InferAction a) noexcept : m_value(std::move(a)) {}

  Action(BenchAction a) noexcept : m_value(std::move(a)) {}

  Action(PopulateAction a) noexcept : m_value(std::move(a)) {}

  Action(HelpAction a) noexcept : m_value(std::move(a)) {}

  Action(DumpCsvAction a) noexcept : m_value(std::move(a)) {}

  Action(ReweightAction a) noexcept : m_value(std::move(a)) {}

  Action(QueryDeviceInfo a) noexcept : m_value(std::move(a)) {}

  Action(MergeDeviceInfo a) noexcept : m_value(std::move(a)) {}

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
    if (std::holds_alternative<DumpCsvAction>(m_value))
      return ActionKind::DumpCsv;
    if (std::holds_alternative<ReweightAction>(m_value))
      return ActionKind::Reweight;
    if (std::holds_alternative<QueryDeviceInfo>(m_value)) 
      return ActionKind::QueryDeviceInfo;
    if (std::holds_alternative<MergeDeviceInfo>(m_value)) 
      return ActionKind::MergeDeviceInfo;
    denox::diag::unreachable();
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

  const ReweightAction &reweight() const noexcept {
    assert(kind() == ActionKind::Reweight);
    return std::get<ReweightAction>(m_value);
  }

  ReweightAction &reweight() noexcept {
    assert(kind() == ActionKind::Reweight);
    return std::get<ReweightAction>(m_value);
  }


  const QueryDeviceInfo &query_device_info() const noexcept {
    assert(kind() == ActionKind::QueryDeviceInfo);
    return std::get<QueryDeviceInfo>(m_value);
  }

  QueryDeviceInfo &query_device_info() noexcept {
    assert(kind() == ActionKind::QueryDeviceInfo);
    return std::get<QueryDeviceInfo>(m_value);
  }

  const MergeDeviceInfo &merge_device_info() const noexcept {
    assert(kind() == ActionKind::MergeDeviceInfo);
    return std::get<MergeDeviceInfo>(m_value);
  }

  MergeDeviceInfo &merge_device_info() noexcept {
    assert(kind() == ActionKind::MergeDeviceInfo);
    return std::get<MergeDeviceInfo>(m_value);
  }

private:
  struct VersionTag {};

  explicit Action(VersionTag) noexcept : m_value(VersionTag{}) {}

private:
  std::variant<CompileAction, InferAction, BenchAction, PopulateAction,
               HelpAction, VersionTag, DumpCsvAction, ReweightAction, QueryDeviceInfo,
               MergeDeviceInfo>
      m_value;
};
