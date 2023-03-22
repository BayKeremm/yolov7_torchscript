cmake_minimum_required(VERSION 3.0 FATAL_ERROR)


project(yolo)
set(CMAKE_CXX_STANDARD 14)
list(APPEND CMAKE_PREFIX_PATH "/home/baykeremm/thesis/libtorch")
list(APPEND CMAKE_PREFIX_PATH "/home/baykeremm/thesis/libtorch/share/cmake")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lm -ldl")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

