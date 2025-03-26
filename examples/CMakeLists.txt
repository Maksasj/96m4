include_directories("../src")
include_directories("../3dparty/stb")

add_executable(96m4_example_0 example_0.cpp)
target_link_libraries(96m4_example_0 stb_image_write)