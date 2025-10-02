#include "compiler/lifeness.hpp"
#include "memory/container/dynamic_bitset.hpp"

namespace denox::compiler {

Lifetimes lifeness(const CanoModel &model) {
  using Graph = CanoModel::Graph;
  const Graph &g = model.graph;

  const std::size_t N = g.upperNodeCount();
  Lifetimes out;
  out.valueLifetimes.resize(N, Lifetime{0, 0});

  // ---- Reachability (from input) ----
  memory::dynamic_bitset reach_in(N, false);
  {
    memory::vector<Graph::NodeHandle> st;
    st.push_back(model.input);
    while (!st.empty()) {
      Graph::NodeHandle n = std::move(st.back());
      st.pop_back();
      const std::size_t id = *n->id();
      if (id >= reach_in.size())
        reach_in.resize(id + 1, false);
      if (reach_in[id])
        continue;
      reach_in[id] = true;
      for (const auto &e : n->outgoing())
        st.push_back(e.dst());
    }
  }

  // ---- Back-reachability (to output) ----
  memory::dynamic_bitset reach_out(N, false);
  {
    memory::vector<Graph::NodeHandle> st;
    st.push_back(model.output);
    while (!st.empty()) {
      Graph::NodeHandle n = std::move(st.back());
      st.pop_back();
      const std::size_t id = *n->id();
      if (id >= reach_out.size())
        reach_out.resize(id + 1, false);
      if (reach_out[id])
        continue;
      reach_out[id] = true;
      for (const auto &e : n->incoming())
        for (const auto &s : e.srcs())
          st.push_back(s);
    }
  }

  // ---- Presence & active sets ----
  memory::dynamic_bitset present(N, false);
  memory::dynamic_bitset active(N, false);
  for (std::size_t i = 0; i < N; ++i) {
    const bool in = (i < reach_in.size() && reach_in[i]);
    const bool out = (i < reach_out.size() && reach_out[i]);
    present[i] = in || out; // id exists in the canonical graph
    active[i] = in && out;  // lies on some input→output path
  }

  // ---- Build node-level arcs (from hyperedges) on the active subgraph ----
  memory::vector<memory::vector<std::size_t>> succ(N);
  memory::vector<std::uint32_t> indeg(N, 0);
  for (std::size_t i = 0; i < N; ++i) {
    if (!active[i])
      continue;
    auto n = g.get(memory::NodeId(i));
    for (const auto &e : n->outgoing()) {
      const std::size_t d = static_cast<std::size_t>(e.dst().id());
      if (!active[d])
        continue;
      for (const auto &s : e.srcs()) {
        const std::size_t sid = static_cast<std::size_t>(s.id());
        if (!active[sid])
          continue;
        succ[sid].push_back(d);
        ++indeg[d];
      }
    }
  }

  // ---- ASAP levels L⁺ via Kahn ----
  memory::vector<std::uint64_t> Lp(N, 0);
  memory::vector<std::size_t> topo;
  topo.reserve(N);
  memory::vector<std::size_t> q;
  q.reserve(N);
  for (std::size_t i = 0; i < N; ++i)
    if (active[i] && indeg[i] == 0)
      q.push_back(i);

  while (!q.empty()) {
    std::size_t u = q.back();
    q.pop_back();
    topo.push_back(u);
    const auto &su = succ[u];
    for (std::size_t j = 0; j < su.size(); ++j) {
      std::size_t v = su[j];
      if (Lp[v] < Lp[u] + 1)
        Lp[v] = Lp[u] + 1;
      if (--indeg[v] == 0)
        q.push_back(v);
    }
  }

  // ---- ALAP latest-start levels L⁻ with deadline T* = max L⁺(sinks) ----
  std::uint64_t Tstar = 0;
  for (std::size_t i = 0; i < N; ++i)
    if (active[i] && succ[i].empty() && Lp[i] > Tstar)
      Tstar = Lp[i];

  memory::vector<std::uint64_t> Lm(N, Tstar);
  for (std::size_t i = 0; i < N; ++i)
    if (!active[i])
      Lm[i] = 0;

  for (std::size_t ti = topo.size(); ti > 0; --ti) {
    std::size_t u = topo[ti - 1];
    if (!active[u])
      continue;
    const auto &su = succ[u];
    if (su.empty()) {
      Lm[u] = Tstar; // sink: latest start equals deadline
    } else {
      std::uint64_t m = std::numeric_limits<std::uint64_t>::max();
      for (std::size_t j = 0; j < su.size(); ++j) {
        const std::size_t v = su[j];
        if (!active[v])
          continue;
        // latest start of u so that v can still start at its latest start (unit
        // edge)
        if (Lm[v] > 0 && Lm[v] - 1 < m)
          m = Lm[v] - 1;
      }
      if (m == std::numeric_limits<std::uint64_t>::max())
        m = Tstar;                     // defensive
      Lm[u] = (m < Lp[u]) ? Lp[u] : m; // clamp to not precede earliest start
    }
  }

  // ---- Lifetimes per id (model B: read-anytime; print/think as half-open)
  // ----
  const auto U64_MAX = std::numeric_limits<std::uint64_t>::max();
  for (std::size_t i = 0; i < N; ++i) {
    if (!present[i]) {
      out.valueLifetimes[i] = {U64_MAX, U64_MAX};
      continue;
    }
    if (!active[i]) {
      out.valueLifetimes[i] = {0, 0};
      continue;
    }

    const std::uint64_t start = Lp[i];
    std::uint64_t end = 0;

    if (succ[i].empty()) {
      // sink value: end at its own latest FINISH ⇒ Lm[i] + 1 (half-open)
      end = Lm[i] + 1;
    } else {
      // non-sink: end at max over consumers' latest FINISH ⇒ Lm[v] + 1
      for (std::size_t j = 0; j < succ[i].size(); ++j) {
        const std::size_t v = succ[i][j];
        if (!active[v])
          continue;
        const std::uint64_t need = Lm[v] + 1;
        if (need > end)
          end = need;
      }
      if (end < start + 1)
        end = start + 1; // at least one level wide
    }

    out.valueLifetimes[i] = {start, end};
  }

  return out;
}

} // namespace denox::compiler
