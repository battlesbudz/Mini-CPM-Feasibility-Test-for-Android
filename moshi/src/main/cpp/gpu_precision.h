#pragma once
#include <ggml-backend.h>
#include <cstdint>
#include <utility>
#include <vector>

// Scoped, thread-local experiment. No process-wide backend settings or weight conversion.
struct GpuPrecisionScope {
    inline static thread_local GpuPrecisionScope* active=nullptr;
    ggml_backend_t backend;
    uint64_t graphs=0, matmuls=0;
    GpuPrecisionScope* previous;
    explicit GpuPrecisionScope(ggml_backend_t target):backend(target),previous(active){active=this;}
    ~GpuPrecisionScope(){active=previous;}
    GpuPrecisionScope(const GpuPrecisionScope&)=delete;
    GpuPrecisionScope& operator=(const GpuPrecisionScope&)=delete;
};
struct GpuPrecisionGraph {
    std::vector<std::pair<ggml_tensor*,ggml_prec>> original;
    GpuPrecisionGraph(ggml_backend_t backend,ggml_cgraph* graph) {
        auto* scope=GpuPrecisionScope::active;
        if(!scope||scope->backend!=backend)return;
        // Reserve before mutations so allocation failure cannot leave a half-modified graph.
        original.reserve(ggml_graph_n_nodes(graph));
        ++scope->graphs;
        for(int i=0;i<ggml_graph_n_nodes(graph);++i){
            auto* node=ggml_graph_node(graph,i);
            if(node->op!=GGML_OP_MUL_MAT)continue;
            // Pinned GGML stores the public precision enum at op_params[0].
            original.emplace_back(node,static_cast<ggml_prec>(node->op_params[0]));
            ggml_mul_mat_set_prec(node,GGML_PREC_F32);++scope->matmuls;
        }
    }
    ~GpuPrecisionGraph(){for(auto [node,precision]:original)ggml_mul_mat_set_prec(node,precision);}
    GpuPrecisionGraph(const GpuPrecisionGraph&)=delete;
    GpuPrecisionGraph& operator=(const GpuPrecisionGraph&)=delete;
};
