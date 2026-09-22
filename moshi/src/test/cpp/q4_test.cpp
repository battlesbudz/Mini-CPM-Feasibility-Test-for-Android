#include "q4_check.h"
#include <ggml-vulkan.h>
#include <cstdlib>
#include <iostream>
int main(int argc,char** argv){
    ggml_backend_t gpu=nullptr;
    try{
        const bool inject=argc==2&&std::string(argv[1])=="--inject-compile-failure";
        if(inject)setenv("MOSHI_TEST_FAIL_Q4_PIPELINE","1",1);
        if(!ggml_backend_vk_get_device_count())throw std::runtime_error("Vulkan unavailable; not skipped");
        gpu=ggml_backend_vk_init(0);if(!gpu)throw std::runtime_error("Vulkan initialization failed");
        try {
            moshi_q4::check(gpu,[](const std::string& s){std::cout<<s<<std::endl;});
            if(inject)throw std::runtime_error("Failure injection did not run");
        }catch(const std::exception& e){
            if(!inject||std::string(e.what())!="Injected Q4 pipeline compile failure")throw;
            std::cout<<"PASS compile exception reached caller without null pipeline dispatch"<<std::endl;
            unsetenv("MOSHI_TEST_FAIL_Q4_PIPELINE");
            // A failed graph invalidates context caches. Android retires its
            // worker after ERROR; test a fresh context, never reuse that graph.
            ggml_backend_free(gpu);gpu=ggml_backend_vk_init(0);
            if(!gpu)throw std::runtime_error("Vulkan reinitialization failed");
            moshi_q4::check(gpu,[](const std::string& s){std::cout<<s<<std::endl;});
            std::cout<<"PASS fresh backend after injected compiler failure"<<std::endl;
        }
        ggml_backend_free(gpu);return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;if(gpu)ggml_backend_free(gpu);return 1;}
}
