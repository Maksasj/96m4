include_directories("./")

FILE(GLOB_RECURSE 96M4_SRC_FILES *.cpp)
add_library(96m4 STATIC ${96M4_SRC_FILES})