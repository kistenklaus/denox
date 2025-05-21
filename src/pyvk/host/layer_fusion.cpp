#include "./layer_fusion.hpp"
#include "pyvk/host/LayerDescription.hpp"
#include "pyvk/host/NetworkDescription.hpp"
#include <algorithm>
#include <print>
#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace pyvk {

enum class TriState {
  Allow,
  Disallow,
  DontCare,
};

class FusionRule {
public:
  virtual ~FusionRule() = default;

  virtual TriState fusable(const LayerDescription &layer0,
                           const LayerDescription &layer1) = 0;
};

class ConvActiFusion final : public FusionRule {
public:
  TriState fusable(const LayerDescription &layer0,
                   const LayerDescription &layer1) override {
    bool x = layer0.type == LayerType::Conv2d &&
             layer1.type == LayerType::Activation;

    return x ? TriState::Allow : TriState::DontCare;
  }
};

class NonTrivalDep final : public FusionRule {
public:
  NonTrivalDep(std::unordered_map<std::string, std::vector<std::string>>
                   outputDependencies)
      : m_outputDependencies(std::move(outputDependencies)) {}

  TriState fusable(const LayerDescription &layer0,
                   [[maybe_unused]] const LayerDescription &layer1) {
    if (m_outputDependencies.at(layer0.name).size() > 1) {
      return TriState::Disallow;
    } else {
      return TriState::DontCare;
    }
  }

private:
  std::unordered_map<std::string, std::vector<std::string>>
      m_outputDependencies;
};

std::vector<DispatchOp> fuseLayers(const NetworkDescription &network) {

  using LayerHandle = std::shared_ptr<LayerDescription>;

  // I hate it myself, but sometimes performance doesn't matter.
  std::unordered_map<std::string, LayerHandle> layerMap;
  std::unordered_map<std::string, std::vector<std::string>> inputDependencies;

  LayerHandle input;
  for (const auto &layer : network.layers()) {
    layerMap[layer->name] = layer;
    std::vector<std::string> &deps = inputDependencies[layer->name];
    switch (layer->type) {
    case LayerType::None:
      // skip
      break;
    case LayerType::Input:
      input = layer;
    case LayerType::Conv2d:
    case LayerType::Activation:
    case LayerType::MaxPool:
    case LayerType::Upsample:
    case LayerType::Output:
      deps.push_back(layer->inputName);
      break;
    case LayerType::Concat:
      deps.push_back(layer->inputName);
      deps.push_back(layer->info.concat.to);
      break;
    }
  }
  // reverse dependency order
  std::unordered_map<std::string, std::vector<std::string>> outputDependencies;
  for (const auto &[layerName, deps] : inputDependencies) {
    for (const auto &dep : deps) {
      outputDependencies[dep].push_back(layerName);
    }
  }

  assert(output != nullptr);

  // Not a linked list! =^).
  struct MyTuple {
    LayerHandle prev;
    LayerHandle curr;
  };

  ConvActiFusion convActiRule;
  NonTrivalDep concatFusion(outputDependencies);
  FusionRule *rules[]{
      &convActiRule, // <- conv + relu is trivially fusable.
      &concatFusion, // <- this rule is sort of implied by the algo, just here for edge cases.
  };
  auto applyRules = [&](const LayerDescription &layer0,
                        const LayerDescription &layer1) -> bool {
    bool possible = false;
    for (const auto &rule : rules) {
      auto tri = rule->fusable(layer0, layer1);
      if (tri == TriState::Allow) {
        possible = true;
      } else if (tri == TriState::Disallow) {
        return false;
      }
    }
    return possible;
  };

  std::stack<MyTuple> topoStack;
  topoStack.push(MyTuple{
      .prev = nullptr,
      .curr = input,
  });
  std::vector<DispatchOp> ops;
  std::stack<std::vector<LayerHandle>> fusedStack;
  std::set<std::string> visited;
  visited.insert("interface");
  while (!topoStack.empty()) {
    if (fusedStack.empty()) {
      fusedStack.push({});
    }

    const auto &x = topoStack.top();
    topoStack.pop();
    const LayerHandle &prev = x.prev;
    const LayerHandle &curr = x.curr;

    if (visited.contains(curr->name)) {
      continue;
    }
    bool allInputDepsAvailable = true;
    for (const auto &dep : inputDependencies[curr->name]) {
      if (!visited.contains(dep)) {
        allInputDepsAvailable = false;
      }
    }
    if (!allInputDepsAvailable) {
      continue; // iam to lazy to fix this right now iam know that it is far
                // from optimal.
    }

    visited.insert(curr->name);

    std::println("Processing pair ({}, {})",
                 prev != nullptr ? prev->name : "null", curr->name);

    bool fuse = prev != nullptr && applyRules(*prev, *curr);

    if (fuse) {
      fusedStack.top().push_back(curr);
      std::println("FUSE");
    } else {
      if (!fusedStack.top().empty()) {
        std::println("Make dispatch");
        ops.push_back(DispatchOp(fusedStack.top()));
      }
      fusedStack.pop();
      fusedStack.push({curr});
    }
    std::vector<std::string> deps = outputDependencies[curr->name];
    for (const auto &dep : deps) {
      std::println("Dependency : {}", dep);
      topoStack.push(MyTuple{.prev = curr, .curr = layerMap.at(dep)});
    }
  }
  while (!fusedStack.empty()) {
    ops.push_back(DispatchOp(fusedStack.top()));
    fusedStack.pop();
  }
  return ops;
}

} // namespace pyvk
