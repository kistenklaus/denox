#!/usr/bin/env python3
from pathlib import Path
import sys

PREFIX = "Model_import_op_"

TEMPLATE = """#pragma once

#include "vkcnn/common/model/import/Model_import_state.inl"
#include "vkcnn/common/model/import/Model_import_tensors.inl"
#include <fmt/format.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>

namespace vkcnn::details {{

static std::vector<Tensor> import_op_{op}(
    [[maybe_unused]] ImportState &state,
    [[maybe_unused]] std::span<const std::optional<Tensor>> inputs,
    [[maybe_unused]] std::size_t outputCount,
    [[maybe_unused]] const std::unordered_map<std::string, Tensor> &attributes,
    [[maybe_unused]] opset_version version, const onnx::NodeProto &node) {{
  throw std::runtime_error(fmt::format(
      "vkcnn: operation {op} is not supported (node = \\"{{}}\\")", node.name()));
}}

}} // namespace vkcnn::details
"""

def op_name_from_filename(name: str) -> str | None:
    if not name.startswith(PREFIX):
        return None
    # Take everything after the prefix, then drop any extension(s)
    rest = name[len(PREFIX):]
    op = rest.split(".", 1)[0]
    return op if op else None

def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory>")
        sys.exit(2)

    d = Path(sys.argv[1])
    if not d.is_dir():
        print(f"Not a directory: {d}")
        sys.exit(1)

    count = 0
    for entry in d.iterdir():
        if not entry.is_file():
            continue
        op = op_name_from_filename(entry.name)
        if not op:
            continue
        try:
            content = TEMPLATE.format(op=op)
            entry.write_text(content, encoding="utf-8")
            count += 1
            print(f"Wrote: {entry.name}  (op={op})")
        except Exception as e:
            print(f"ERROR writing {entry}: {e}")

    print(f"\nDone. Overwrote {count} file(s).")

if __name__ == "__main__":
    main()
