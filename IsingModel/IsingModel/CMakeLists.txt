project(IsingModel CXX CUDA)

include_directories(/usr/local/cuda-11.1/include)
add_compile_options(--std c++17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../bin/Debug)
else()
	set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../bin/Release)
endif()
add_executable(IsingModel IsingModel.cu)

target_compile_options(IsingModel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_86,code=sm_86>)
set_target_properties(IsingModel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)