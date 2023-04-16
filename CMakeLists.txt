cmake_minimum_required(VERSION 3.0 FATAL_ERROR)


project(yolo)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED PATHS "/home/garfield/thesis/libtorch" )
find_package(OpenCV REQUIRED)
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lm -ldl")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

