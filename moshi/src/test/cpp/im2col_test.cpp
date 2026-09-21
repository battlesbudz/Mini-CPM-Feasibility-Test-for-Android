#include "im2col_check.h"
#include <ggml-cpu.h>
#ifdef MOSHI_TEST_VULKAN
#include <ggml-vulkan.h>
#endif
#include <iostream>
int main() {
    auto cpu=ggml_backend_cpu_init();
    ggml_backend_t gpu=nullptr;
    try {
        if(!cpu) throw std::runtime_error("CPU unavailable");
        ggml_backend_cpu_set_n_threads(cpu,4);
        auto log=[](const std::string& s){std::cout<<s<<std::endl;};
        std::cout<<"CPU scalar-oracle regression"<<std::endl;
        moshi_im2col::check(cpu,log);
#ifdef MOSHI_TEST_VULKAN
        if(ggml_backend_vk_get_device_count()==0) throw std::runtime_error("Vulkan device unavailable (not a skip)");
        gpu=ggml_backend_vk_init(0);
        if(!gpu) throw std::runtime_error("Vulkan initialization failed");
        std::cout<<"Vulkan scalar-oracle regression"<<std::endl;
        moshi_im2col::check(gpu,log);
#endif
    } catch(const std::exception& e) {
        std::cerr<<e.what()<<std::endl;
        if(gpu)ggml_backend_free(gpu);if(cpu)ggml_backend_free(cpu);return 1;
    }
    if(gpu)ggml_backend_free(gpu);ggml_backend_free(cpu);return 0;
}
