#!/usr/bin/env python3
"""Fetch pinned dependencies and make a reproducible Android integration copy."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
PINS = {
    "moshi.cpp": ("Codes4Fun/moshi.cpp", "f1fabbd14a506076d4d0a9755811598220ee9e13"),
    "ggml-moshi": ("Codes4Fun/ggml", "8cf09e9cd3c227ecb42aefc544c820b6c63a28f3"),
    "sentencepiece-moshi": ("google/sentencepiece", "17d7580d6407802f85855d2cc9190634e2c95624"),
    "Vulkan-Headers-moshi": ("KhronosGroup/Vulkan-Headers", "19725e4d48082fe78e26622b15d3080ccd54112b"),
}

def checked_replace(path, before, after, count=1):
    text = path.read_text()
    if text.count(before) != count:
        raise RuntimeError(f"Pinned integration anchor changed: {path}: {before[:60]}")
    path.write_text(text.replace(before, after))

def main():
    VENDOR.mkdir(exist_ok=True)
    for name, (repo, pin) in PINS.items():
        dest = VENDOR / name
        if not dest.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", f"https://github.com/{repo}.git", str(dest)], check=True)
        if subprocess.check_output(["git", "-C", str(dest), "status", "--porcelain"]):
            raise RuntimeError(f"Refusing to overwrite changes in {dest}")
        subprocess.run(["git", "-C", str(dest), "checkout", "--detach", pin], check=True)
    # Keep upstream checkout pristine; all backend fixes are reproducible overlays.
    ggml_adapted = VENDOR / "ggml-moshi-android"
    if ggml_adapted.exists():
        shutil.rmtree(ggml_adapted)
    shutil.copytree(VENDOR / "ggml-moshi", ggml_adapted, ignore=shutil.ignore_patterns(".git"))
    shader = ggml_adapted / "src/ggml-vulkan/vulkan-shaders/im2col.comp"
    shutil.copyfile(ROOT / "moshi/src/main/cpp/shaders/im2col.comp", shader)
    backend = ggml_adapted / "src/ggml-vulkan/ggml-vulkan.cpp"
    # IM2COL 1D/2D alone uses descriptor writes with fixed 64-thread dispatch.
    # Leave IM2COL_3D and all unrelated BDA pipelines untouched.
    for variant in ("im2col_f32", "im2col_f32_f16_rte", "im2col_f32_f16"):
        checked_replace(backend, variant + " ## bda ## _len", variant + "_len")
        checked_replace(backend, variant + " ## bda ## _data", variant + "_data")
    checked_replace(backend,
        "sizeof(vk_op_im2col_push_constants), {512, 1, 1}, { device->subgroup_size }",
        "sizeof(vk_op_im2col_push_constants), {64, 1, 1}, {}", 3)
    checked_replace(backend,
        "if (ctx->device->shader_int64 && ctx->device->buffer_device_address) {\n            // buffer device address path doesn't use dst buffer",
        "if (op == GGML_OP_IM2COL_3D && ctx->device->shader_int64 && ctx->device->buffer_device_address) {\n            // Only 3D still uses buffer device addresses; 1D/2D needs the full descriptor range.")
    # The scalar small tile covers exactly one 32x32 virtual warp. On an
    # 8-lane Vulkan device, BLOCK_SIZE=16 with WARP=8 creates a second virtual
    # warp outside that tile, causing overlapping writes and shared-memory OOB.
    # Keep cooperative-matrix layouts and >=16-lane devices (including Adreno)
    # unchanged. Logical scalar lanes do not use hardware subgroup operations.
    checked_replace(backend,
        "s_warptile = { subgroup_size_16, 32, 32, 16, 32, 32, 2, tm_s, tn_s, tk_s, subgroup_size_8 };",
        "s_warptile = { subgroup_size_16, 32, 32, 16, 32, 32, 2, tm_s, tn_s, tk_s, device->coopmat_support ? subgroup_size_8 : subgroup_size_16 };")
    checked_replace(backend,
        "s_warptile = { subgroup_size_16, 32, 32, 16, 32, 32, 2, 2, 2, 1, subgroup_size_8 };",
        "s_warptile = { subgroup_size_16, 32, 32, 16, 32, 32, 2, 2, 2, 1, subgroup_size_16 };")
    # Build 11: Adreno rejected mul_mat_vec_q4_k_f32_f32 at first inference.
    # Use an explicitly bounded scalar Q4_K path; keep all other quants, matrix
    # kernels, expert routing and integer-dot variants unchanged.
    shaders = ggml_adapted / "src/ggml-vulkan/vulkan-shaders"
    shutil.copyfile(ROOT / "moshi/src/main/cpp/shaders/mul_mat_vec_q4_k_portable.comp",
                    shaders / "mul_mat_vec_q4_k_portable.comp")
    checked_replace(shaders / "vulkan-shaders-gen.cpp", "    // flash attention\n", '''    string_to_spv("moshi_q4_k_f32", "mul_mat_vec_q4_k_portable.comp", {});
    string_to_spv("moshi_q4_k_f16", "mul_mat_vec_q4_k_portable.comp", {{"MOSHI_B_F16", "1"}});

    // flash attention
''')
    for btype in ("f32", "f16"):
        old = f'''            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_{btype}_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_{btype}_f32", arr_dmmv_q4_k_{btype}_f32_len[reduc16], arr_dmmv_q4_k_{btype}_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {{rm_kq, 1, 1}}, {{wg_size_subgroup16, rm_kq, i+1}}, 1, true, use_subgroups16, force_subgroup_size16);'''
        new = f'''            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_{btype}_f32[w][GGML_TYPE_Q4_K][i], "moshi_q4_k_portable_{btype}", moshi_q4_k_{btype}_len, moshi_q4_k_{btype}_data, "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {{1, 1, 1}}, {{64, 1, i+1}}, 1, false, false, 0);'''
        checked_replace(backend, old, new)
    # The pinned runtime waits on compiler futures without get(): driver errors
    # are silently discarded, then the caller uses an uncompiled pipeline.
    # Drain every future, propagate the first error, and release the compile slot
    # even on exceptions. This also prevents a later compile waiting forever.
    checked_replace(backend, "    GGML_ASSERT(parameter_count > 0);", '''    struct CompileSlot {
        ~CompileSlot() {
            { std::lock_guard<std::mutex> guard(compile_count_mutex); --compile_count; }
            compile_count_cond.notify_all();
        }
    } compile_slot;
#ifdef MOSHI_VULKAN_TEST_FAILURE
    if (pipeline->name == "moshi_q4_k_portable_f32" && getenv("MOSHI_TEST_FAIL_Q4_PIPELINE"))
        throw std::runtime_error("Injected Q4 pipeline compile failure");
#endif
    GGML_ASSERT(parameter_count > 0);''')
    checked_replace(backend, '''    {
        std::lock_guard<std::mutex> guard(compile_count_mutex);
        assert(compile_count > 0);
        compile_count--;
    }
    compile_count_cond.notify_all();''', '')
    checked_replace(backend, '''    for (auto &c : compiles) {
        c.wait();
    }''', '''    std::exception_ptr compile_error;
    for (auto &c : compiles) {
        try { c.get(); } catch (...) { if (!compile_error) compile_error = std::current_exception(); }
    }
    if (compile_error) std::rethrow_exception(compile_error);''')
    checked_replace(backend,
        '''        std::cerr << "ggml_vulkan: Compute pipeline creation failed for " << pipeline->name << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;''',
        '''        std::cerr << "ggml_vulkan: Compute pipeline creation failed for " << pipeline->name << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << " spec=";
        for (auto value : specialization_constants) std::cerr << value << ',';
        std::cerr << " required_subgroup=" << required_subgroup_size
                  << " full_subgroups=" << require_full_subgroups << std::endl;
        throw;''')
    adapted = VENDOR / "moshi-android-src"
    if adapted.exists():
        shutil.rmtree(adapted)
    shutil.copytree(VENDOR / "moshi.cpp", adapted, ignore=shutil.ignore_patterns(".git"))
    # Correct byte sizing in both loaders. No on-device weight conversion.
    checked_replace(adapted / "src/context.h", "std::vector<char*> data(nbytes);", "std::vector<char> data(nbytes);")
    checked_replace(adapted / "src/loader.h", "std::vector<char*> data;", "std::vector<char> data;")
    checked_replace(adapted / "src/loader.h", "if ( data.size() < nbytes ) data.resize( nbytes );", "data.resize(std::min<size_t>(nbytes, 8 * 1024 * 1024));")
    checked_replace(adapted / "src/loader.h", '''            int64_t r = fread(data.data(), nbytes, 1, f);
            if (r != 1) {
                printf("failed to read tensor %s\\n", name.c_str());
                exit(-1);
            }
            ggml_backend_tensor_set(tensor, data.data(), 0, nbytes);''', '''            for (size_t done = 0; done < nbytes;) {
                const size_t chunk = std::min(data.size(), nbytes - done);
                if (fread(data.data(), 1, chunk, f) != chunk) {
                    fclose(f);
                    throw std::runtime_error("Truncated GGUF tensor: " + name);
                }
                ggml_backend_tensor_set(tensor, data.data(), done, chunk);
                done += chunk;
            }''')
    checked_replace(adapted / "src/loader.h", 'auto f = fopen( filename.c_str(), "rb" );', 'auto f = fopen( filename.c_str(), "rb" );\n        if (!f) throw std::runtime_error("Cannot open GGUF weights");')
    checked_replace(adapted / "src/loader.h", "buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);", 'buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);\n        if (!buffer) throw std::runtime_error("GGUF weight allocation failed");', 2)
    # Fail explicitly when a selected backend cannot execute the graph.
    checked_replace(adapted / "src/context.h", "ggml_backend_graph_compute( backend, gf );", '''for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            auto node = ggml_graph_node(gf, i);
            if (!ggml_backend_supports_op(backend, node))
                throw std::runtime_error(std::string("Unsupported backend operation: ") + ggml_op_name(node->op));
        }
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Backend graph execution failed");''', 2)
    checked_replace(adapted / "src/context.h", "assert( buffer );", 'if (!buffer) throw std::runtime_error("Graph buffer allocation failed");', 6)
    checked_replace(adapted / "src/context.h", '''        assert(false);
    }

    ggml_tensor * fill( NE ne, float value )''', '''        throw std::runtime_error("fill requires a backend");
    }

    ggml_tensor * fill( NE ne, float value )''')
    # Opt-in tracing is implemented in the app; normal runs retain whole-graph execution.
    for source in ("context.h", "loader.h", "torch.h", "moshi/models/compression.h", "moshi/models/tts.h", "replay_ops.h"):
        path = adapted / "src" / source
        text = '#include "tensor_trace.h"\n' + path.read_text().replace("ggml_backend_tensor_set(", "moshi_trace_tensor_set(")
        if source == "context.h":
            text = text.replace("ggml_backend_graph_compute(backend, gf)", "moshi_trace_compute(backend, gf)")
        path.write_text(text)
    assets = ROOT / "moshi/src/main/assets"
    for name in PINS:
        license_file = VENDOR / name / "LICENSE"
        if not license_file.exists():
            license_file = VENDOR / name / "LICENSE.txt"
        if not license_file.exists():
            license_file = VENDOR / name / "LICENSE.md"
        if license_file.exists():
            shutil.copyfile(license_file, assets / (name + "-LICENSE.txt"))
        extra_licenses = VENDOR / name / "LICENSES"
        if extra_licenses.is_dir():
            for extra in extra_licenses.glob("*.txt"):
                shutil.copyfile(extra, assets / (name + "-" + extra.name))
    print("Prepared pinned Moshi Android integration")

if __name__ == "__main__":
    main()
