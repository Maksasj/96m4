include_directories("../src")
include_directories("../3dparty/stb")

add_executable(hello hello.cpp)

add_executable(96m4_example_0 example_0.cpp)
target_link_libraries(96m4_example_0 stb_image_write)
target_link_libraries(96m4_example_0 96m4)

add_executable(pong_demo pong_demo.cpp)
target_link_libraries(pong_demo stb_image_write)
target_link_libraries(pong_demo 96m4)

add_executable(pong_demo_1 pong_demo_1.cpp)
target_link_libraries(pong_demo_1 stb_image_write)
target_link_libraries(pong_demo_1 96m4)

add_executable(pong_demo_small pong_demo_small.cpp)
target_link_libraries(pong_demo_small stb_image_write)
target_link_libraries(pong_demo_small 96m4)

add_executable(image_demo image_demo.cpp)
target_link_libraries(image_demo stb_image)
target_link_libraries(image_demo stb_image_write)
target_link_libraries(image_demo 96m4)

add_executable(image_demo_small image_demo_small.cpp)
target_link_libraries(image_demo_small stb_image)
target_link_libraries(image_demo_small stb_image_write)
target_link_libraries(image_demo_small 96m4)


add_executable(math_demo math_demo.cpp)
target_link_libraries(math_demo stb_image)
target_link_libraries(math_demo stb_image_write)
target_link_libraries(math_demo 96m4)