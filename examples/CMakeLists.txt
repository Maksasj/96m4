include_directories("../src")
include_directories("../3dparty/stb")

add_executable(96m4_example_0 example_0.cpp)
target_link_libraries(96m4_example_0 stb_image_write)

add_executable(96m4_example_1 example_1.cpp)
target_link_libraries(96m4_example_1 stb_image_write)