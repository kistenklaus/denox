#include "./layer_fusion.hpp"
#include "pyvk/host/LayerDescription.hpp"
#include "pyvk/host/NetworkDescription.hpp"
#include <algorithm>
#include <print>
#include <set>
#include <stack>
#include <unordered_map>
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

class ConvActiFusion : public FusionRule {
public:
  TriState fusable(const LayerDescription &layer0,
                   const LayerDescription &layer1) override {
    bool x = layer0.type == LayerType::Conv2d &&
             layer1.type == LayerType::Activation;

    return x ? TriState::Allow : TriState::DontCare;
  }
};

class DisallowConcatFusion : public FusionRule {
public:
  DisallowConcatFusion(
      std::span<const std::shared_ptr<LayerDescription>> layers) {
    for (const auto &layer : layers) {
      if (layer->type == LayerType::Concat) {
        m_concatDependencies.insert(layer->info.concat.to);
        m_concatDependencies.insert(layer->name);
      }
    }
  }
  TriState fusable(const LayerDescription &layer0,
                   const LayerDescription &layer1) override {
    if (m_concatDependencies.contains(layer0.name)) {
      return TriState::Disallow;
    } else {
      return TriState::DontCare;
    }
  }

private:
  std::set<std::string> m_concatDependencies;
};

std::vector<DispatchOp> fuseLayers(const NetworkDescription &network) {

  ConvActiFusion convActiRule;
  DisallowConcatFusion concatFusion(network.layers());
  FusionRule *rules[]{
      &convActiRule, //
      &concatFusion, //
  };
  using LayerHandle = std::shared_ptr<LayerDescription>;

  // I hate it myself, but sometimes performance doesn't matter.
  std::unordered_map<std::string, LayerHandle> layerMap;

  LayerHandle output;
  for (const auto &layer : network.layers()) {
    std::println("Register {}", layer->name);
    if (layer->type == LayerType::Concat) {
      std::println("To {}", layer->info.concat.to);
    }
    layerMap[layer->name] = layer;
    if (layer->type == LayerType::Output) {
      output = layer;
    }
  }
  assert(output != nullptr);

  // Not a linked list! =^).
  struct MyTuple {
    LayerHandle
        prev; // <- this is actually the furthest node (depth in the CNN dag)
    LayerHandle next; // <- this is the closest node to the input.
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
      .next = output,
  });
  std::vector<DispatchOp> ops;
  std::stack<std::vector<LayerHandle>> fusedStack;
  std::set<std::string> visited;
  while (!topoStack.empty()) {
    if (fusedStack.empty()) {
      fusedStack.push({});
    }

    const auto &x = topoStack.top();
    topoStack.pop();
    const LayerHandle &prev = x.prev;
    const LayerHandle &next = x.next;

    if (visited.contains(next->name)) {
      continue;
    }
    visited.insert(next->name);


    bool fuse = prev != nullptr && applyRules(*next, *prev);

    if (fuse) {
      fusedStack.top().push_back(next);
      std::println("FUSION");
    } else {
      if (!fusedStack.top().empty()) {
        std::ranges::reverse(fusedStack.top());
        ops.push_back(DispatchOp(fusedStack.top()));
      }
      fusedStack.pop();
      fusedStack.push({next});
    }

    // Enqueue dependencies
    switch (next->type) {
    case LayerType::None:
      // skip (treated like a noop)
      break;
    case LayerType::Input:
      break;
    case LayerType::Conv2d:
      topoStack.push(
          MyTuple{.prev = next, .next = layerMap.at(next->inputName)});
      break;
    case LayerType::MaxPool:
      topoStack.push(
          MyTuple{.prev = next, .next = layerMap.at(next->inputName)});
      break;
    case LayerType::Upsample:
      topoStack.push(
          MyTuple{.prev = next, .next = layerMap.at(next->inputName)});
      break;
    case LayerType::Concat: {
      auto x = layerMap.at(next->inputName);
      auto y = layerMap.at(next->info.concat.to);
      topoStack.push(MyTuple{.prev = next, .next = y});
      topoStack.push(MyTuple{.prev = next, .next = x});
      break;
    }
    case LayerType::Activation:
      topoStack.push(
          MyTuple{.prev = next, .next = layerMap.at(next->inputName)});
      break;
    case LayerType::Output:
      topoStack.push(
          MyTuple{.prev = next, .next = layerMap.at(next->inputName)});
      break;
    }
  }
  std::ranges::reverse(ops);
  return ops;
}

} // namespace pyvk
