include_directories("../../src")
include_directories("../../3dparty/stb")

add_executable(96m4_perceptron perceptron.cpp)
target_link_libraries(96m4_perceptron stb_image_write)