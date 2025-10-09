#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>        // containers, string, etc.
#include <string>
#include <vector>
#include <optional>
#include <fstream>
#include <stdexcept>
#include <cctype>

#include "denox/compiler.hpp"

namespace py = pybind11;

// ---------- helpers: parsing ----------

static denox::VulkanApiVersion parse_target_env(const std::string &s) {
    if (s == "vulkan1.0" || s == "1.0") return denox::VulkanApiVersion::Vulkan_1_0;
    if (s == "vulkan1.1" || s == "1.1") return denox::VulkanApiVersion::Vulkan_1_1;
    if (s == "vulkan1.2" || s == "1.2") return denox::VulkanApiVersion::Vulkan_1_2;
    if (s == "vulkan1.3" || s == "1.3") return denox::VulkanApiVersion::Vulkan_1_3;
    if (s == "vulkan1.4" || s == "1.4") return denox::VulkanApiVersion::Vulkan_1_4;
    throw std::invalid_argument("invalid target-env: " + s);
}

static denox::DataType parse_dtype(const std::string &s) {
    if (s == "f16" || s == "float16") return denox::DataType::Float16;
    if (s == "f32" || s == "float32") return denox::DataType::Float32;
    if (s == "u8"  || s == "uint8")   return denox::DataType::Uint8;
    if (s == "i8"  || s == "int8")    return denox::DataType::Int8;
    if (s == "auto"|| s == "?")       return denox::DataType::Auto;
    throw std::invalid_argument("invalid dtype: " + s);
}

static denox::Layout parse_layout(const std::string &s) {
    if (s == "HWC")   return denox::Layout::HWC;
    if (s == "CHW")   return denox::Layout::CHW;
    if (s == "CHWC8") return denox::Layout::CHWC8;
    throw std::invalid_argument("invalid layout: " + s);
}

static denox::Storage parse_storage(const std::string &s) {
    if (s == "SSBO" || s == "buffer" || s == "storage-buffer")
        return denox::Storage::StorageBuffer;
    throw std::invalid_argument("invalid storage: " + s);
}

static denox::FeatureState parse_feature_state(const py::object &o) {
    if (o.is_none()) return denox::FeatureState::Enable; // default like CLI
    if (py::isinstance<py::bool_>(o))  return o.cast<bool>() ? denox::FeatureState::Enable
                                                             : denox::FeatureState::Disable;
    auto s = o.cast<std::string>();
    for (auto &c : s) c = char(::tolower(unsigned(c)));
    if (s == "require") return denox::FeatureState::Require;
    if (s == "enable")  return denox::FeatureState::Enable;
    if (s == "disable") return denox::FeatureState::Disable;
    throw std::invalid_argument("feature state must be 'require'|'enable'|'disable' or bool");
}

// Store names here so `const char*` lifetimes cover the compile() call.
struct NameArena {
    std::vector<std::string> names;
    const char* intern(std::string s) {
        names.emplace_back(std::move(s));
        return names.back().c_str();
    }
};

static bool is_digits(const std::string &s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c){ return std::isdigit(c)!=0; });
}

// Accept: int, None/'?' (infer), "Name", "Name=123", "123"
static denox::Extent parse_extent_obj(const py::object &o, NameArena &arena) {
    denox::Extent e{nullptr, 0};

    if (o.is_none()) return e;

    if (py::isinstance<py::int_>(o)) {
        auto v = o.cast<long long>();
        if (v < 0) v = -v;
        e.value = static_cast<unsigned int>(v);
        return e;
    }

    std::string s = o.cast<std::string>();
    // '?' -> infer
    auto trim = [](std::string &t){
        while (!t.empty() && std::isspace((unsigned char)t.front())) t.erase(t.begin());
        while (!t.empty() && std::isspace((unsigned char)t.back()))  t.pop_back();
    };
    trim(s);
    if (s == "?") return e;

    auto pos = s.find('=');
    if (pos == std::string::npos) {
        if (is_digits(s)) {
            e.value = static_cast<unsigned int>(std::stoul(s));
        } else {
            e.name = arena.intern(s);
        }
        return e;
    } else {
        std::string name = s.substr(0, pos);
        std::string rhs  = s.substr(pos+1);
        trim(name); trim(rhs);
        e.name = name.empty() ? nullptr : arena.intern(name);
        if (!rhs.empty()) e.value = static_cast<unsigned int>(std::stoul(rhs));
        return e;
    }
}

// Accept: None, "H:W:C", or a 3-seq like (H, W, C). Elements per parse_extent_obj.
static denox::Shape parse_shape_obj(const py::object &o, NameArena &arena) {
    denox::Shape sh{};
    if (o.is_none()) return sh;

    if (py::isinstance<py::sequence>(o) && !py::isinstance<py::str>(o)) {
        auto seq = o.cast<py::sequence>();
        if (py::len(seq) != 3) throw std::invalid_argument("shape sequence must have length 3");
        sh.height   = parse_extent_obj(seq[0], arena);
        sh.width    = parse_extent_obj(seq[1], arena);
        sh.channels = parse_extent_obj(seq[2], arena);
        return sh;
    }

    // string "H:W:C"
    std::string s = o.cast<std::string>();
    auto colon1 = s.find(':'), colon2 = std::string::npos;
    if (colon1 != std::string::npos) colon2 = s.find(':', colon1+1);
    if (colon1 == std::string::npos || colon2 == std::string::npos)
        throw std::invalid_argument("shape must be 'H:W:C' or a 3-tuple/list");

    auto H = s.substr(0, colon1);
    auto W = s.substr(colon1+1, colon2-colon1-1);
    auto C = s.substr(colon2+1);
    sh.height   = parse_extent_obj(py::str(H), arena);
    sh.width    = parse_extent_obj(py::str(W), arena);
    sh.channels = parse_extent_obj(py::str(C), arena);
    return sh;
}

/* ---------- helpers: same as you have now (parse_* and NameArena) ---------- */
/* ... keep your parse_target_env / parse_dtype / parse_layout / parse_storage /
   parse_feature_state / NameArena / parse_extent_obj / parse_shape_obj ...     */

/* ---------- small wrapper class for the compiled blob ---------- */
struct Program {
    std::shared_ptr<std::string> blob;  // own the bytes

    explicit Program(std::string data)
        : blob(std::make_shared<std::string>(std::move(data))) {}

    std::size_t size() const { return blob ? blob->size() : 0; }

    py::bytes to_bytes() const {
        if (!blob) return py::bytes();
        return py::bytes(blob->data(), blob->size());
    }

    void save(py::object pathlike) const {
        if (!blob) throw std::runtime_error("empty program");
        const std::string path = py::str(pathlike);
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("failed to open file: " + path);
        f.write(blob->data(), static_cast<std::streamsize>(blob->size()));
        if (!f) throw std::runtime_error("failed to write file: " + path);
    }

    std::string repr() const {
        return "<denox.Program size=" + std::to_string(size()) + " bytes>";
    }
};

/* ---------- core compile helper (builds CompileOptions and calls C) ---------- */
struct CompileResultBytes {
    std::string dnx;  // copy into C++-owned memory so we can free the C result
};

static CompileResultBytes do_compile_to_bytes(
    const std::string &model,
    std::optional<std::string> device,
    const std::string &target_env,
    py::object input_shape, py::object output_shape,
    const std::string &input_type, const std::string &output_type,
    const std::string &input_layout, const std::string &output_layout,
    const std::string &input_storage, const std::string &output_storage,
    py::object coopmat, py::object fusion, py::object memory_concat,
    bool spirv_debug_info, bool spirv_non_semantic_debug_info,
    bool spirv_optimize, bool spirv_skip_compilation,
    bool verbose, bool quiet, bool summarize,
    py::object cwd)
{
    denox::CompileOptions opt{};
    NameArena arena;

    opt.dnxVersion = 0;
    opt.srcType    = denox::SrcType::Auto;
    opt.heuristic  = denox::Heuristic::MemoryBandwidth;

    // features
    opt.features.coopmat       = parse_feature_state(coopmat);
    opt.features.fusion        = parse_feature_state(fusion);
    opt.features.memory_concat = parse_feature_state(memory_concat);

    // IO
    opt.inputDescription.shape   = parse_shape_obj(input_shape,  arena);
    opt.outputDescription.shape  = parse_shape_obj(output_shape, arena);
    opt.inputDescription.dtype   = parse_dtype(input_type);
    opt.outputDescription.dtype  = parse_dtype(output_type);
    opt.inputDescription.layout  = parse_layout(input_layout);
    opt.outputDescription.layout = parse_layout(output_layout);
    opt.inputDescription.storage = parse_storage(input_storage);
    opt.outputDescription.storage= parse_storage(output_storage);

    // device
    opt.device.deviceName = device ? arena.intern(*device) : nullptr;
    opt.device.apiVersion = parse_target_env(target_env);

    // spirv
    opt.spirvOptions.debugInfo            = spirv_debug_info;
    opt.spirvOptions.nonSemanticDebugInfo = spirv_non_semantic_debug_info;
    opt.spirvOptions.optimize             = spirv_optimize;
    opt.spirvOptions.skipCompilation      = spirv_skip_compilation;

    // misc
    opt.cwd       = cwd.is_none() ? nullptr : arena.intern(cwd.cast<std::string>());
    opt.verbose   = verbose;
    opt.quite     = quiet;
    opt.summarize = summarize;

    denox::CompilationResult res{};
    int rc;
    {
        py::gil_scoped_release release;
        rc = denox::compile(model.c_str(), &opt, &res);
    }

    struct Guard {
        denox::CompilationResult &r;
        ~Guard() { denox::destroy_compilation_result(&r); }
    } guard{res};

    if (rc < 0) {
        const char* msg = res.message ? res.message : "denox::compile failed";
        throw std::runtime_error(std::string(msg));
    }

    if (!res.dnx || res.dnxSize == 0) return {std::string()};

    // copy into std::string we own
    return { std::string(static_cast<const char*>(res.dnx), res.dnxSize) };
}

/* ---------- Python-callable wrappers ---------- */

// 1) returns Program object
static Program py_compile_program(
    const std::string &model,
    std::optional<std::string> device,
    const std::string &target_env,
    py::object input_shape, py::object output_shape,
    const std::string &input_type, const std::string &output_type,
    const std::string &input_layout, const std::string &output_layout,
    const std::string &input_storage, const std::string &output_storage,
    py::object coopmat, py::object fusion, py::object memory_concat,
    bool spirv_debug_info, bool spirv_non_semantic_debug_info,
    bool spirv_optimize, bool spirv_skip_compilation,
    bool verbose, bool quiet, bool summarize,
    py::object cwd)
{
    auto r = do_compile_to_bytes(
        model, device, target_env, input_shape, output_shape,
        input_type, output_type, input_layout, output_layout,
        input_storage, output_storage,
        coopmat, fusion, memory_concat,
        spirv_debug_info, spirv_non_semantic_debug_info,
        spirv_optimize, spirv_skip_compilation,
        verbose, quiet, summarize, cwd);
    return Program(std::move(r.dnx));
}

// 2) returns raw bytes
static py::bytes py_compile_bytes(
    const std::string &model,
    std::optional<std::string> device,
    const std::string &target_env,
    py::object input_shape, py::object output_shape,
    const std::string &input_type, const std::string &output_type,
    const std::string &input_layout, const std::string &output_layout,
    const std::string &input_storage, const std::string &output_storage,
    py::object coopmat, py::object fusion, py::object memory_concat,
    bool spirv_debug_info, bool spirv_non_semantic_debug_info,
    bool spirv_optimize, bool spirv_skip_compilation,
    bool verbose, bool quiet, bool summarize,
    py::object cwd)
{
    auto r = do_compile_to_bytes(
        model, device, target_env, input_shape, output_shape,
        input_type, output_type, input_layout, output_layout,
        input_storage, output_storage,
        coopmat, fusion, memory_concat,
        spirv_debug_info, spirv_non_semantic_debug_info,
        spirv_optimize, spirv_skip_compilation,
        verbose, quiet, summarize, cwd);
    return py::bytes(r.dnx.data(), r.dnx.size());
}

PYBIND11_MODULE(_denox, m) {
    m.doc() = "denox Python binding";

    py::class_<Program>(m, "Program")
        .def("save",      &Program::save,      py::arg("path"),
             "Write the program bytes to a file.")
        .def("to_bytes",  &Program::to_bytes,
             "Return the program as Python bytes.")
        .def_property_readonly("size", &Program::size)
        .def("__len__",   &Program::size)
        .def("__bytes__", &Program::to_bytes)
        .def("__repr__",  &Program::repr);

    auto add_sig = [&](auto name, auto fn){
        m.def(
            name, fn,
            py::arg("model"),
            py::kw_only(),
            py::arg("device") = py::none(),
            py::arg("target_env") = "vulkan1.1",
            py::arg("input_shape") = py::none(),
            py::arg("output_shape") = py::none(),
            py::arg("input_type") = "auto",
            py::arg("output_type") = "auto",
            py::arg("input_layout") = "HWC",
            py::arg("output_layout") = "HWC",
            py::arg("input_storage") = "SSBO",
            py::arg("output_storage") = "SSBO",
            py::arg("coopmat") = py::none(),        // 'require'|'enable'|'disable' or bool
            py::arg("fusion") = py::none(),
            py::arg("memory_concat") = py::none(),
            py::arg("spirv_debug_info") = false,
            py::arg("spirv_non_semantic_debug_info") = false,
            py::arg("spirv_optimize") = false,
            py::arg("spirv_skip_compilation") = false,
            py::arg("verbose") = false,
            py::arg("quiet") = false,
            py::arg("summarize") = false,
            py::arg("cwd") = py::none()
        );
    };

    add_sig("compile",        &py_compile_program); // returns Program
    add_sig("compile_bytes",  &py_compile_bytes);   // returns bytes
}
