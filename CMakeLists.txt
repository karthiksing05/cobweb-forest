cmake_minimum_required(VERSION 3.18...3.30)
project(cobweb_forest LANGUAGES CXX)
# The hybrid Cobweb of cobweb-private (commit in csrc/UPSTREAM_COMMIT), forest iteration:
# NaN-masked continuous queries, integer node uids, expanded() and leaves_only mixtures.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(_core NB_DOMAIN cobweb_forest csrc/src/cobweb_hybrid.cpp csrc/src/cobweb_hybrid_node.cpp
                    csrc/src/cobweb_hybrid_tree.cpp csrc/src/helper.cpp)
target_include_directories(_core PRIVATE csrc/include third_party/eigen third_party/rapidjson_include)
target_compile_features(_core PRIVATE cxx_std_17)
set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/cobweb_forest)
