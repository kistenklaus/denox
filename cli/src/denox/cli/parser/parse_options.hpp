#include "denox/cli/io/IOEndpoint.hpp"
#include "denox/cli/parser/artefact.hpp"
#include "denox/cli/parser/lex/tokens/token.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/memory/container/hashmap.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/string.hpp"
#include <cstdint>
#include <span>

uint32_t parse_help(std::span<const Token> tokens, bool *help);

uint32_t parse_spirv_optimize(std::span<const Token> t, bool *v);

uint32_t parse_spirv_non_semantic_debug_info(std::span<const Token> t, bool *v);

uint32_t parse_spirv_debug_info(std::span<const Token> t, bool *v);

uint32_t parse_verbose(std::span<const Token> t, bool *v);

uint32_t parse_quiet(std::span<const Token> t, bool *v);

uint32_t parse_feature_coopmat(std::span<const Token> t, bool *v);

uint32_t parse_feature_fusion(std::span<const Token> t, bool *v);

uint32_t parse_feature_memory_concat(std::span<const Token> t, bool *v);

uint32_t parse_output(std::span<const Token> tokens,
                      std::optional<IOEndpoint> *out);

uint32_t
parse_device(std::span<const Token> tokens,
             denox::memory::optional<denox::memory::string> *deviceName);

uint32_t parse_target_env(std::span<const Token> tokens,
                          denox::ApiVersion *target_env);

uint32_t parse_database(std::span<const Token> tokens,
                        std::optional<DbArtefact> *db);

uint32_t
parse_shape(std::span<const Token> tokens,
            denox::memory::hash_map<denox::memory::string,
                                    denox::compiler::InterfaceTensorDescriptor>
                &tensorDescriptors);

uint32_t parse_storage(
    std::span<const Token> tokens,
    denox::memory::hash_map<denox::memory::string,
                            denox::compiler::InterfaceTensorDescriptor>
        &tensorDescriptors);

uint32_t
parse_format(std::span<const Token> tokens,
             denox::memory::hash_map<denox::memory::string,
                                     denox::compiler::InterfaceTensorDescriptor>
                 &tensorDescriptors);

uint32_t
parse_type(std::span<const Token> tokens,
           denox::memory::hash_map<denox::memory::string,
                                   denox::compiler::InterfaceTensorDescriptor>
               &tensorDescriptors);

uint32_t
parse_use_descriptor_sets(std::span<const Token> tokens,
                          denox::compiler::DescriptorPolicies *policies);

uint32_t parse_assume(
    std::span<const Token> tokens,
    denox::memory::hash_map<denox::memory::string, int64_t> &assumptions);

uint32_t parse_specialize(
    std::span<const Token> tokens,
    denox::memory::hash_map<denox::memory::string, int64_t> &assumptions);

uint32_t parse_samples(std::span<const Token> tokens, uint32_t *samples);

uint32_t parse_relative_error(std::span<const Token> tokens,
                              float* relative_error);

uint32_t parse_input(std::span<const Token> tokens,
                     denox::memory::optional<IOEndpoint> *endpoint);

uint32_t parse_optimizationLevel(std::span<const Token> tokens,
                                uint32_t *optimizationLevel);
