cmake_minimum_required(VERSION 3.16)
project(rflux VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(RFLUX_BUILD_TESTS "Build C and C++ tests" ON)
option(RFLUX_BUILD_EXAMPLES "Build example programs" ON)
option(RFLUX_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

set(RFLUX_CORE_SOURCES
    src/qflux/window.cpp
    src/qflux/indicators.cpp
    src/qflux/stats.cpp
    src/qflux/ranking.cpp
    src/regimelab/hmm.cpp
    src/regimelab/regime.cpp
    src/percolab/percolation.cpp
    src/percolab/network.cpp)

add_library(rflux_core STATIC ${RFLUX_CORE_SOURCES})
target_include_directories(rflux_core
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(rflux_core PUBLIC cxx_std_17)
if(NOT WIN32)
    target_link_libraries(rflux_core PUBLIC m)
endif()
set_target_properties(rflux_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)

add_library(rflux SHARED src/rflux_capi.cpp)
add_library(rflux::rflux ALIAS rflux)
target_link_libraries(rflux PRIVATE rflux_core)
target_include_directories(rflux
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(rflux PROPERTIES
    OUTPUT_NAME rflux
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
if(WIN32)
    target_compile_definitions(rflux PRIVATE RFLUX_BUILDING_DLL)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(rflux_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(rflux PRIVATE -Wall -Wextra -Wpedantic)
    if(RFLUX_WARNINGS_AS_ERRORS)
        target_compile_options(rflux_core PRIVATE -Werror)
        target_compile_options(rflux PRIVATE -Werror)
    endif()
elseif(MSVC)
    target_compile_options(rflux_core PRIVATE /W4)
    target_compile_options(rflux PRIVATE /W4)
    if(RFLUX_WARNINGS_AS_ERRORS)
        target_compile_options(rflux_core PRIVATE /WX)
        target_compile_options(rflux PRIVATE /WX)
    endif()
endif()

if(RFLUX_BUILD_EXAMPLES)
    add_executable(rflux_example_c examples/c_basic.c)
    target_link_libraries(rflux_example_c PRIVATE rflux)
    set_target_properties(rflux_example_c PROPERTIES OUTPUT_NAME example_c_basic)

    add_executable(rflux_example_cpp examples/cpp_research.cpp)
    target_link_libraries(rflux_example_cpp PRIVATE rflux_core)
    target_include_directories(rflux_example_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    set_target_properties(rflux_example_cpp PROPERTIES OUTPUT_NAME example_cpp_research)
endif()

if(RFLUX_BUILD_TESTS)
    enable_testing()
    add_executable(rflux_test_c tests/c/test_capi.c)
    target_link_libraries(rflux_test_c PRIVATE rflux)
    set_target_properties(rflux_test_c PROPERTIES OUTPUT_NAME test_capi)
    add_test(NAME rflux_c_abi COMMAND rflux_test_c)

    add_executable(rflux_test_cpp tests/cpp/test_cpp.cpp)
    target_link_libraries(rflux_test_cpp PRIVATE rflux_core)
    target_include_directories(rflux_test_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    set_target_properties(rflux_test_cpp PROPERTIES OUTPUT_NAME test_cpp)
    add_test(NAME rflux_cpp_core COMMAND rflux_test_cpp)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS rflux
    EXPORT rfluxTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS rflux_core
    EXPORT rfluxTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/rflux DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT rfluxTargets
    FILE rfluxTargets.cmake
    NAMESPACE rflux::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rflux)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rfluxConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rfluxConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rflux)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rfluxConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rfluxConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rfluxConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rflux)
