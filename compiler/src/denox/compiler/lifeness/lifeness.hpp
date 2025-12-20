#pragma once

#include "compiler/ir/CanoModel.hpp"
#include "compiler/ir/Lifetimes.hpp"
namespace denox::compiler {

Lifetimes lifeness(const CanoModel &model);

}
