project(CPUTest)

add_compile_options(-std=c++17)
add_compile_options(-march=native)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../bin/Debug)
else()
	set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../bin/Release)
endif()

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED STATIC)
find_package(glfw3 REQUIRED)

add_executable(CPUTest CPUTest.cpp)