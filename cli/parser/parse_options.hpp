#include "denox/common/types.hpp"
#include "denox/compiler.hpp"
#include "io/IOEndpoint.hpp"
#include "parser/artefact.hpp"
#include "parser/lex/tokens/token.hpp"
#include <cstdint>
#include <span>

uint32_t parse_spirv_optimize(std::span<const Token> tokens, bool *flag);

uint32_t parse_spirv_optimize(std::span<const Token> t, bool *v);

uint32_t parse_spirv_non_semantic_debug_info(std::span<const Token> t, bool *v);

uint32_t parse_spirv_debug_info(std::span<const Token> t, bool *v);

uint32_t parse_verbose(std::span<const Token> t, bool *v);

uint32_t parse_quiet(std::span<const Token> t, bool *v);

uint32_t parse_feature_coopmat(std::span<const Token> t,
                               denox::FeatureState *v);

uint32_t parse_feature_fusion(std::span<const Token> t, denox::FeatureState *v);

uint32_t parse_feature_memory_concat(std::span<const Token> t,
                                     denox::FeatureState *v);

uint32_t parse_output(std::span<const Token> tokens,
                      std::optional<IOEndpoint> *out);

uint32_t parse_device(std::span<const Token> tokens, const char **device);

uint32_t parse_target_env(std::span<const Token> tokens,
                          denox::VulkanApiVersion *target_env);

uint32_t parse_input_layout(
    std::span<const Token> tokens,
    std::unordered_map<std::string, denox::BufferDescription> &inputs);

uint32_t parse_output_layout(
    std::span<const Token> tokens,
    std::unordered_map<std::string, denox::BufferDescription> &outputs);

uint32_t parse_input_storage(
    std::span<const Token> tokens,
    std::unordered_map<std::string, denox::BufferDescription> &inputs);

uint32_t parse_output_storage(
    std::span<const Token> tokens,
    std::unordered_map<std::string, denox::BufferDescription> &outputs);

uint32_t parse_input_type(
    std::span<const Token> tokens,
    std::unordered_map<std::string, denox::BufferDescription> &inputs);

uint32_t parse_output_type(
    std::span<const Token> tokens,
    std::unordered_map<std::string, denox::BufferDescription> &inputs);

uint32_t parse_input_shape(std::span<const Token> tokens, denox::Shape *shape);

uint32_t
parse_input_shape(std::span<const Token> tokens,
                  std::unordered_map<std::string, denox::BufferDescription>
                      &input_descriptors);
uint32_t
parse_output_shape(std::span<const Token> tokens,
                   std::unordered_map<std::string, denox::BufferDescription>
                       &output_descriptors);

uint32_t parse_database(std::span<const Token> tokens,
                        std::optional<DbArtefact> *db);

uint32_t parse_assume(std::span<const Token> tokens,
                      std::vector<denox::OptimizationAssumption> &assumptions);
