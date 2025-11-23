#pragma once
#include "compiler/ir/populate/ImplDb.hpp"
#include "io/fs/Path.hpp"
#include "memory/container/shared_ptr.hpp"
#include "shaders/compiler/ShaderBinary.hpp"
namespace denox::compiler {

namespace db::details {

struct ShaderBinary {
  std::vector<uint32_t> source;
};

enum class Access {
  ReadOnly,
  WriteOnly,
  ReadWrite,
};

struct TensorBinding {
  unsigned int set;
  unsigned int binding;
  Access access;

  uint64_t byteSize;
  unsigned int minAlignment;
};

struct Timing {
  uint64_t samples = 0;
  uint64_t latency_ns = 0;
  uint64_t std_derivation_ns = 0;
};

struct ComputeDispatch {
  uint32_t binaryId;
  uint32_t workgroupCountX;
  uint32_t workgroupCountY;
  uint32_t workgroupCountZ;
  std::vector<uint8_t> pushConstant;
  std::vector<TensorBinding> bindings;
  Timing time;
};

struct Operation {
  std::string shaderName;
  uint32_t pattern;
  uint32_t config;
  uint64_t hash;
  std::vector<uint32_t> dispatches;
};

struct Db {
  io::Path path;
  std::vector<ShaderBinary> binaries;
  std::vector<ComputeDispatch> dispatches;
  std::vector<Operation> operations;

  uint64_t addShaderBinary(const denox::compiler::ShaderBinary &shaderBinary);

  uint64_t
  addComputeDispatch(const denox::compiler::DbComputeDispatch &dispatch);

  uint64_t addOp(const denox::compiler::DbOp &op);

  void write_back();
};

}; // namespace db::details

class Db {
public:
  static Db open(const io::Path &path);

  uint64_t addShaderBinary(const denox::compiler::ShaderBinary &shaderBinary);

  uint64_t
  addComputeDispatch(const denox::compiler::DbComputeDispatch &dispatch);

  uint64_t addOp(const denox::compiler::DbOp &op);

  const db::details::Db *get() const {
    assert(m_db != nullptr);
    return m_db.get();
  }

  void write_back() {
    m_db->write_back();
  }

private:
  Db(std::shared_ptr<db::details::Db> db) : m_db(std::move(db)) {}
  memory::shared_ptr<db::details::Db> m_db;
};

} // namespace denox::compiler
