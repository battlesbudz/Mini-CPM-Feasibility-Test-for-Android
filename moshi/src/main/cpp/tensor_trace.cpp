#include "tensor_trace.h"
#include "ggml-impl.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <map>
namespace {
std::string jsonString(const std::string& text) {
    std::ostringstream s;s<<'"';
    for(unsigned char c:text){if(c=='"'||c=='\\')s<<'\\'<<c;else if(c<32)s<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else s<<c;}
    s<<'"';return s.str();
}
constexpr size_t maxTensorBytes=64*1024*1024;
constexpr int maxSamples=4096;
struct Snapshot {
    std::vector<double> values;
    std::string skipped;
    int64_t elements=0;
};
Snapshot readTensor(ggml_tensor* t) {
    Snapshot result;result.elements=ggml_nelements(t);
    if(ggml_nbytes(t)>maxTensorBytes){result.skipped="tensor_over_64_MiB";return result;}
    size_t width=0;
    switch(t->type){case GGML_TYPE_F32:case GGML_TYPE_I32:width=4;break;case GGML_TYPE_F16:case GGML_TYPE_BF16:width=2;break;default:result.skipped="unsupported_sample_type";return result;}
    std::vector<unsigned char> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t,bytes.data(),0,bytes.size());
    int64_t n=std::min<int64_t>(maxSamples,result.elements);
    for(int64_t i=0;i<n;++i){
        int64_t k=n==1?0:i*(result.elements-1)/(n-1),rest=k;size_t offset=0;
        for(int d=0;d<4;++d){offset+=(rest%t->ne[d])*t->nb[d];rest/=t->ne[d];}
        if(offset+width>bytes.size())throw std::runtime_error("Trace tensor stride out of bounds");
        double value;
        if(t->type==GGML_TYPE_F32){float v;memcpy(&v,bytes.data()+offset,4);value=v;}
        else if(t->type==GGML_TYPE_I32){int32_t v;memcpy(&v,bytes.data()+offset,4);value=v;}
        else if(t->type==GGML_TYPE_F16){ggml_fp16_t v;memcpy(&v,bytes.data()+offset,2);value=ggml_fp16_to_fp32(v);}
        else {ggml_bf16_t v;memcpy(&v,bytes.data()+offset,2);value=ggml_bf16_to_fp32(v);}
        result.values.push_back(value);
    }
    return result;
}
struct Difference {
    bool mismatch=false;int bad=0,nonfinite=0,firstBad=-1;double firstCpu=0,firstTested=0;double error=0,refEnergy=0,testEnergy=0,maxAbs=0;
    Difference(const Snapshot& cpu,const Snapshot& gpu,bool integer){
        if(cpu.values.size()!=gpu.values.size())throw std::runtime_error("Trace sample shape mismatch");
        for(size_t i=0;i<cpu.values.size();++i){double a=cpu.values[i],b=gpu.values[i];
            if(!std::isfinite(a)||!std::isfinite(b)){
                ++nonfinite;
                // Matching signed infinities are legitimate attention-mask values.
                // NaNs and unmatched infinities are always discrepancies.
                if(!(std::isinf(a)&&a==b)){++bad;if(firstBad<0){firstBad=int(i);firstCpu=a;firstTested=b;}}
                continue;
            }
            double e=std::abs(a-b);maxAbs=std::max(maxAbs,e);error+=e*e;refEnergy+=a*a;testEnergy+=b*b;
            if(e>(integer?0:1e-4+1e-3*std::abs(a))){++bad;if(firstBad<0){firstBad=int(i);firstCpu=a;firstTested=b;}}
        }mismatch=bad>0;
    }
};
}
struct TensorTrace::Impl {
    ggml_backend_t tested,reference;std::function<void(const std::string&)> progress;
    std::ofstream log;std::string phase="initialization",first="null";
    std::map<std::string,std::string> firstByStage;
    bool stageDifferent=false;
    int graphs=0,nodes=0,differences=0,skipped=0,uploads=0,uploadFailures=0;uint64_t uploadBytes=0;
    void line(const std::string& s){log<<s<<'\n';log.flush();if(!log)throw std::runtime_error("Trace log write failed");}
    std::string location(int graph,int node,ggml_tensor* tensor)const{
        return "\"stage\":"+jsonString(phase)+",\"graph\":"+std::to_string(graph)+",\"node\":"+std::to_string(node)+",\"op\":"+jsonString(ggml_op_name(tensor->op))+",\"name\":"+jsonString(tensor->name);
    }
    bool compare(ggml_tensor* cpu,ggml_tensor* gpu,int graph,int node,const std::string& role){
        for(int d=0;d<4;++d)if(cpu->ne[d]!=gpu->ne[d]||cpu->nb[d]!=gpu->nb[d])throw std::runtime_error("Trace graph layout mismatch");
        if(cpu->type!=gpu->type)throw std::runtime_error("Trace tensor type mismatch");
        auto a=readTensor(cpu),b=readTensor(gpu);
        Difference diff(a,b,cpu->type==GGML_TYPE_I32);
        std::ostringstream s;s<<std::setprecision(10)<<"{"<<location(graph,node,gpu)<<",\"role\":"<<jsonString(role)
            <<",\"type\":"<<jsonString(ggml_type_name(gpu->type))<<",\"elements\":"<<b.elements<<",\"bytes\":"<<ggml_nbytes(gpu)<<",\"shape\":[";
        for(int d=0;d<4;++d){if(d)s<<',';s<<gpu->ne[d];}s<<"],\"strides\":[";
        for(int d=0;d<4;++d){if(d)s<<',';s<<gpu->nb[d];}s<<"],\"opParams\":[";
        for(size_t d=0;d<sizeof(gpu->op_params)/sizeof(gpu->op_params[0]);++d){if(d)s<<',';s<<gpu->op_params[d];}
        s<<"],\"sampleCount\":"<<a.values.size();
        if(!a.skipped.empty()||!b.skipped.empty()){++skipped;s<<",\"skipped\":"<<jsonString(a.skipped.empty()?b.skipped:a.skipped);}
        else{
            double n=std::max<size_t>(1,a.values.size());
            s<<",\"cpuRms\":"<<std::sqrt(diff.refEnergy/n)<<",\"testedRms\":"<<std::sqrt(diff.testEnergy/n)
                <<",\"maxAbsError\":"<<diff.maxAbs<<",\"nrmse\":"<<std::sqrt(diff.error/std::max(1e-30,diff.refEnergy))
                <<",\"outsideTolerance\":"<<diff.bad<<",\"nonfiniteSamples\":"<<diff.nonfinite<<",\"mismatch\":"<<(diff.mismatch?"true":"false");
            if(diff.firstBad>=0){
                // Strings preserve NaN/infinity without emitting invalid JSON numbers.
                s<<",\"firstBadLogicalIndex\":"<<(a.values.size()==1?0:int64_t(diff.firstBad)*(a.elements-1)/(a.values.size()-1))
                    <<",\"firstCpuValue\":"<<jsonString(std::to_string(diff.firstCpu))<<",\"firstTestedValue\":"<<jsonString(std::to_string(diff.firstTested));
            }
        }
        s<<'}';line(s.str());
        if(diff.mismatch && role=="output") {++differences;if(first=="null")first=s.str();if(!firstByStage.count(phase))firstByStage[phase]=s.str();}
        return diff.mismatch;
    }
};
namespace {thread_local TensorTrace::Impl* active=nullptr;}
TensorTrace::TensorTrace(const std::string& dir,ggml_backend_t tested,ggml_backend_t reference,std::function<void(const std::string&)> progress):impl(new Impl){
    if(active)throw std::runtime_error("Nested tensor tracing is not supported");
    impl->tested=tested;impl->reference=reference;impl->progress=std::move(progress);
    impl->log.open(dir+"/tensor-trace.jsonl");if(!impl->log)throw std::runtime_error("Cannot create tensor trace");active=impl.get();
}
TensorTrace::~TensorTrace(){active=nullptr;}
void TensorTrace::stage(const std::string& name){impl->phase=name;impl->stageDifferent=false;impl->progress("Tensor trace: "+name);}
std::string TensorTrace::summary()const{
    std::ostringstream s;s<<"{\"graphs\":"<<impl->graphs<<",\"nodes\":"<<impl->nodes<<",\"differentNodes\":"<<impl->differences
        <<",\"skippedComparisons\":"<<impl->skipped<<",\"verifiedUploadChunks\":"<<impl->uploads<<",\"verifiedUploadBytes\":"<<impl->uploadBytes
        <<",\"uploadFailures\":"<<impl->uploadFailures<<",\"firstDivergence\":"<<impl->first;
    s<<",\"firstDivergenceByStage\":{";bool comma=false;
    for(const auto& item:impl->firstByStage){if(comma)s<<',';comma=true;s<<jsonString(item.first)<<':'<<item.second;}s<<'}';
    s
        <<",\"sampling\":\"Up to 4096 logical elements per tensor; absolute tolerance 0.0001 plus relative tolerance 0.001. Inputs of each graph are copied from the tested backend to CPU. Weights and explicit uploads are verified byte-for-byte separately. Per-node execution changes fusion and synchronization.\"}";return s.str();
}
void moshi_trace_tensor_set(ggml_tensor* tensor,const void* data,size_t offset,size_t size){
    ggml_backend_tensor_set(tensor,data,offset,size);
    if(!active)return;
    // Verify complete uploads in bounded chunks, including model weights, inputs and states.
    std::vector<unsigned char> copy(std::min<size_t>(size,1024*1024));
    for(size_t done=0;done<size;done+=copy.size()){
        size_t n=std::min(copy.size(),size-done);ggml_backend_tensor_get(tensor,copy.data(),offset+done,n);
        ++active->uploads;active->uploadBytes+=n;
        if(memcmp(copy.data(),static_cast<const unsigned char*>(data)+done,n)){
            ++active->uploadFailures;
            active->line("{\"event\":\"upload_mismatch\",\"stage\":"+jsonString(active->phase)+",\"name\":"+jsonString(tensor->name)+",\"offset\":"+std::to_string(offset+done)+"}");
            throw std::runtime_error("Tensor upload readback mismatch");
        }
    }
}
ggml_status moshi_trace_compute(ggml_backend_t backend,ggml_cgraph* graph){
    if(!active||backend!=active->tested)return ggml_backend_graph_compute(backend,graph);
    auto& t=*active;int number=t.graphs++;
    if(t.graphs>64||t.nodes+graph->n_nodes>20000)throw std::runtime_error("Tensor trace graph budget exceeded");
    t.progress("Tensor trace: "+t.phase+" graph "+std::to_string(number));
    auto copy=ggml_backend_graph_copy(t.reference,graph);
    if(!copy.buffer)throw std::runtime_error("Cannot allocate CPU graph reference");
    struct Cleanup {struct ggml_backend_graph_copy copy;~Cleanup(){ggml_backend_graph_copy_free(copy);}} cleanup{copy};
    for(int i=0;i<graph->n_nodes;++i){
        auto gpu=graph->nodes[i],cpu=copy.graph->nodes[i];
        if(!ggml_backend_supports_op(t.reference,cpu))throw std::runtime_error("CPU reference operation unsupported");
        // Check sources BEFORE dispatch, so in-place writes cannot conceal an input difference.
        // Only retain source samples at the first discrepant output in each graph.
        std::vector<Snapshot> inputsCpu,inputsGpu;
        bool inspect=!t.stageDifferent;
        if(inspect)for(int k=0;k<GGML_MAX_SRC;++k){if(!cpu->src[k])break;inputsCpu.push_back(readTensor(cpu->src[k]));inputsGpu.push_back(readTensor(gpu->src[k]));}
        t.line("{\"event\":\"dispatch\","+t.location(number,i,gpu)+"}");
        auto cpuView=ggml_graph_view(copy.graph,i,i+1),gpuView=ggml_graph_view(graph,i,i+1);
        if(ggml_backend_graph_compute(t.reference,&cpuView)!=GGML_STATUS_SUCCESS||ggml_backend_graph_compute(backend,&gpuView)!=GGML_STATUS_SUCCESS)
            throw std::runtime_error("Traced graph execution failed");
        ggml_backend_synchronize(backend);ggml_backend_synchronize(t.reference);++t.nodes;
        bool mismatch=t.compare(cpu,gpu,number,i,"output");
        if(inspect&&mismatch){
            t.stageDifferent=true;
            for(size_t k=0;k<inputsCpu.size();++k){Difference d(inputsCpu[k],inputsGpu[k],cpu->src[k]->type==GGML_TYPE_I32);
                std::ostringstream s;s<<std::setprecision(10)<<"{\"event\":\"first_divergence_input\","<<t.location(number,i,gpu)<<",\"source\":"<<k
                    <<",\"nameOfSource\":"<<jsonString(gpu->src[k]->name)<<",\"type\":"<<jsonString(ggml_type_name(gpu->src[k]->type))
                    <<",\"sampleCount\":"<<inputsCpu[k].values.size()<<",\"maxAbsError\":"<<d.maxAbs<<",\"outsideTolerance\":"<<d.bad
                    <<",\"skipped\":"<<jsonString(inputsCpu[k].skipped)<<'}';t.line(s.str());
            }
        }
        if(i%50==0)t.progress("Tensor trace: "+t.phase+" node "+std::to_string(i)+" / "+std::to_string(graph->n_nodes));
    }
    return GGML_STATUS_SUCCESS;
}
