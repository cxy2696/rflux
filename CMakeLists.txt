cmake_minimum_required(VERSION 3.26)
project(rflux VERSION 0.9.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(RFLUX_BUILD_TESTS "Build C and C++ tests" ON)
option(RFLUX_BUILD_EXAMPLES "Build example programs" ON)
option(RFLUX_BUILD_PYTHON "Build the pybind11 Python extension" OFF)
option(RFLUX_PYTHON_PACKAGE "Install the Python wheel layout" OFF)
option(RFLUX_STABLE_ABI "Build the Python extension against the CPython stable ABI" OFF)
option(RFLUX_BUILD_GTEST "Build the GoogleTest suite when GoogleTest is available" ON)
option(RFLUX_FETCH_GTEST "Download GoogleTest when it is not installed" OFF)
option(RFLUX_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(RFLUX_ENABLE_IPO "Enable interprocedural optimization when supported" ON)

include(CheckIPOSupported)
set(RFLUX_IPO_SUPPORTED OFF)
if(RFLUX_ENABLE_IPO AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    check_ipo_supported(RESULT RFLUX_IPO_RESULT OUTPUT RFLUX_IPO_OUTPUT)
    if(RFLUX_IPO_RESULT)
        set(RFLUX_IPO_SUPPORTED ON)
        message(STATUS "rflux: interprocedural optimization enabled")
    else()
        message(STATUS "rflux: interprocedural optimization unavailable: ${RFLUX_IPO_OUTPUT}")
    endif()
endif()

function(rflux_enable_ipo target)
    if(RFLUX_IPO_SUPPORTED)
        set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endfunction()

if(SKBUILD)
    set(RFLUX_BUILD_PYTHON ON)
    set(RFLUX_PYTHON_PACKAGE ON)
    set(RFLUX_BUILD_TESTS OFF)
    set(RFLUX_BUILD_EXAMPLES OFF)
    if(SKBUILD_SABI_VERSION)
        set(RFLUX_STABLE_ABI ON)
    endif()
endif()

set(RFLUX_CORE_SOURCES
    src/qflux/window.cpp
    src/qflux/indicators.cpp
    src/qflux/stats.cpp
    src/qflux/ranking.cpp
    src/qflux/performance.cpp
    src/regimelab/hmm.cpp
    src/regimelab/regime.cpp
    src/regimelab/alignment.cpp
    src/regimelab/attribution.cpp
    src/percolab/percolation.cpp
    src/percolab/network.cpp
    src/percolab/detector.cpp
    src/percolab/factory.cpp
    src/risk/pretrade.cpp
    src/risk/tail_dependence.cpp
    src/factorlab/factors.cpp
    src/factorlab/evaluation.cpp
    src/factorlab/backtest.cpp
    src/factorlab/factory.cpp
    src/portfoliolab/hrp.cpp
    src/portfoliolab/risk.cpp
    src/portfoliolab/advice.cpp
    src/portfoliolab/factory.cpp
    src/execution/almgren_chriss.cpp
    src/execution/microstructure.cpp
    src/execution/factory.cpp
    src/regimelab/detector.cpp
    src/regimelab/factory.cpp
    src/mllab/huber.cpp
    src/mllab/robust.cpp
    src/mllab/factory.cpp)

find_package(Threads REQUIRED)

add_library(rflux_core STATIC ${RFLUX_CORE_SOURCES})
target_include_directories(rflux_core
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(rflux_core PUBLIC cxx_std_17)
target_link_libraries(rflux_core PUBLIC Threads::Threads)
if(NOT WIN32)
    target_link_libraries(rflux_core PUBLIC m)
endif()
set_target_properties(rflux_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
rflux_enable_ipo(rflux_core)

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
rflux_enable_ipo(rflux)
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

if(RFLUX_BUILD_PYTHON)
    if(RFLUX_STABLE_ABI)
        find_package(Python REQUIRED COMPONENTS Interpreter Development.Module
                     Development.SABIModule)
        if(Python_VERSION VERSION_LESS 3.12)
            message(FATAL_ERROR "RFLUX_STABLE_ABI requires CPython 3.12 or newer")
        endif()
        find_package(pybind11 3.2 CONFIG QUIET)
        if(NOT pybind11_FOUND)
            set(pybind11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/share/cmake/pybind11")
            find_package(pybind11 3.2 CONFIG REQUIRED)
        endif()
        pybind11_add_module(rflux_native STABLE_ABI src/python/bindings.cpp)
    else()
        find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
        find_package(pybind11 3.0 CONFIG QUIET)
        if(NOT pybind11_FOUND)
            set(pybind11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/share/cmake/pybind11")
            find_package(pybind11 3.0 CONFIG REQUIRED)
        endif()
        pybind11_add_module(rflux_native MODULE src/python/bindings.cpp)
    endif()
    target_link_libraries(rflux_native PRIVATE rflux_core)
    target_include_directories(rflux_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(rflux_native PRIVATE
        RFLUX_VERSION_STRING="${PROJECT_VERSION}"
        RFLUX_NATIVE_STABLE_ABI=$<IF:$<BOOL:${RFLUX_STABLE_ABI}>,1,0>)
    set_target_properties(rflux_native PROPERTIES
        OUTPUT_NAME "_native"
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)
    rflux_enable_ipo(rflux_native)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(rflux_native PRIVATE -Wall -Wextra -Wpedantic)
        if(RFLUX_WARNINGS_AS_ERRORS)
            target_compile_options(rflux_native PRIVATE -Werror)
        endif()
    elseif(MSVC)
        target_compile_options(rflux_native PRIVATE /W4)
        if(RFLUX_WARNINGS_AS_ERRORS)
            target_compile_options(rflux_native PRIVATE /WX)
        endif()
    endif()
endif()

if(RFLUX_BUILD_EXAMPLES)
    add_executable(rflux_example_c examples/c_basic.c)
    target_link_libraries(rflux_example_c PRIVATE rflux)
    if(NOT WIN32)
        target_link_libraries(rflux_example_c PRIVATE m)
    endif()
    set_target_properties(rflux_example_c PROPERTIES OUTPUT_NAME example_c_basic)
    rflux_enable_ipo(rflux_example_c)

    add_executable(rflux_example_cpp examples/cpp_research.cpp)
    target_link_libraries(rflux_example_cpp PRIVATE rflux_core)
    target_include_directories(rflux_example_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    set_target_properties(rflux_example_cpp PROPERTIES OUTPUT_NAME example_cpp_research)
    rflux_enable_ipo(rflux_example_cpp)
endif()

if(RFLUX_BUILD_TESTS)
    enable_testing()
    add_executable(rflux_test_c tests/c/test_capi.c)
    target_link_libraries(rflux_test_c PRIVATE rflux)
    if(NOT WIN32)
        target_link_libraries(rflux_test_c PRIVATE m)
    endif()
    set_target_properties(rflux_test_c PROPERTIES OUTPUT_NAME test_capi)
    rflux_enable_ipo(rflux_test_c)
    add_test(NAME rflux_c_abi COMMAND rflux_test_c)

    if(RFLUX_BUILD_GTEST)
        find_package(GTest QUIET)
        if(NOT GTest_FOUND AND RFLUX_FETCH_GTEST)
            include(FetchContent)
            set(GTEST_CMAKE_CXX_STANDARD 17 CACHE STRING "" FORCE)
            set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
            set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
            FetchContent_Declare(
                googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG v1.15.2
                GIT_SHALLOW TRUE)
            FetchContent_MakeAvailable(googletest)
        endif()
        if(TARGET GTest::gtest_main)
            add_executable(rflux_test_gtest tests/cpp/test_core_gtest.cpp)
            target_link_libraries(rflux_test_gtest PRIVATE rflux_core GTest::gtest_main)
            rflux_enable_ipo(rflux_test_gtest)
            target_include_directories(rflux_test_gtest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
            add_test(NAME rflux_cpp_gtest COMMAND rflux_test_gtest)
        else()
            message(WARNING
                "GoogleTest was not found. Install it, or configure with "
                "-DRFLUX_FETCH_GTEST=ON to download it; skipping gtest target.")
        endif()
    endif()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(RFLUX_PYTHON_PACKAGE)
    # A wheel owns only the importable extension; the C ABI library is a
    # separate C/C++ artifact installed by the normal CMake path.
    if(RFLUX_BUILD_PYTHON)
        install(TARGETS rflux_native
            LIBRARY DESTINATION rflux
            RUNTIME DESTINATION rflux)
    endif()
else()
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
endif()
