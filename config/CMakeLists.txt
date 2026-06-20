## Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

cmake_minimum_required(VERSION 3.15)

# ── Single source of truth for version ───────────────────────────────────────
# manifest.json "version" must match this. Build scripts derive the artifact
# name from here. Bump all three in one place.
project(yolo26_people_analytics_plugin VERSION 1.0.0 LANGUAGES CXX)
set(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")

# Static MSVC runtime: keeps the plugin .dll self-contained on Windows
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

# ── Generated artifacts ───────────────────────────────────────────────────────
# plugin_version.h  → included by C++ sources that need compile-time version
# version_info.json → bundled in the release artifact zip/tar
string(TIMESTAMP _BUILD_TIMESTAMP UTC)
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/plugin_version.h.in"
    "${CMAKE_BINARY_DIR}/include/plugin_version.h"
    @ONLY
)
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/version_info.json.in"
    "${CMAKE_BINARY_DIR}/version_info.json"
    @ONLY
)

set(metadataSdkVersion "6.0.6.41837")
set(metadataSdkDir "" CACHE PATH
    "Path to unpacked VMS Metadata SDK zip. Tested with version ${metadataSdkVersion}.")

if(metadataSdkDir STREQUAL "")
    message(FATAL_ERROR "metadataSdkDir cache variable is undefined.")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

# Download OpenCV automatically using Conan
if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
   message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
   file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/master/conan.cmake"
       "${CMAKE_BINARY_DIR}/conan.cmake")
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

# Patch conan.cmake to support MSVC 1944+ (VS 2022 17.4+)
# Override _get_msvc_ide_version function to handle MSVC 1940+
function(_get_msvc_ide_version result)
    set(${result} "" PARENT_SCOPE)
    if(NOT MSVC_VERSION VERSION_LESS 1400 AND MSVC_VERSION VERSION_LESS 1500)
        set(${result} 8 PARENT_SCOPE)
    elseif(NOT MSVC_VERSION VERSION_LESS 1500 AND MSVC_VERSION VERSION_LESS 1600)
        set(${result} 9 PARENT_SCOPE)
    elseif(NOT MSVC_VERSION VERSION_LESS 1600 AND MSVC_VERSION VERSION_LESS 1700)
        set(${result} 10 PARENT_SCOPE)
    elseif(NOT MSVC_VERSION VERSION_LESS 1700 AND MSVC_VERSION VERSION_LESS 1800)
        set(${result} 11 PARENT_SCOPE)
    elseif(NOT MSVC_VERSION VERSION_LESS 1800 AND MSVC_VERSION VERSION_LESS 1900)
        set(${result} 12 PARENT_SCOPE)
    elseif(NOT MSVC_VERSION VERSION_LESS 1900 AND MSVC_VERSION VERSION_LESS 1910)
        set(${result} 14 PARENT_SCOPE)
    elseif(NOT MSVC_VERSION VERSION_LESS 1910 AND MSVC_VERSION VERSION_LESS 1920)
        set(${result} 15 PARENT_SCOPE)
    elseif(NOT MSVC_VERSION VERSION_LESS 1920 AND MSVC_VERSION VERSION_LESS 1930)
        set(${result} 16 PARENT_SCOPE)
    elseif(NOT MSVC_VERSION VERSION_LESS 1930 AND MSVC_VERSION VERSION_LESS 1950)
        # Support MSVC 1940-1949 (VS 2022 17.4+)
        set(${result} 17 PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Conan: Unknown MSVC compiler version [${MSVC_VERSION}]")
    endif()
endfunction()

# Check for Conan executable before using it
find_program(CONAN_CMD conan)
if(NOT CONAN_CMD)
    message(FATAL_ERROR 
        "Conan executable not found!\n"
        "Please install Conan using one of the following methods:\n"
        "  1. pip install conan (for Conan 1.x)\n"
        "  2. pip install conan (for Conan 2.x, recommended)\n"
        "  3. Or download from https://conan.io/downloads\n"
        "\n"
        "After installation, make sure 'conan' is in your PATH.\n"
        "You can verify by running 'conan --version' in a terminal."
    )
endif()
message(STATUS "Found Conan: ${CONAN_CMD}")

if(WIN32)
    set(CONAN_SETTINGS
        "compiler=Visual Studio"
        "compiler.version=17"
        "compiler.runtime=MT"
        "compiler.toolset=v142"
    )
elseif(UNIX)
    # Auto-detect GCC or Clang version for Conan 1.x settings
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        string(REGEX MATCH "^([0-9]+)" _ignored "${CMAKE_CXX_COMPILER_VERSION}")
        set(CONAN_SETTINGS
            "compiler=gcc"
            "compiler.version=${CMAKE_MATCH_1}"
            "compiler.libcxx=libstdc++11"
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        string(REGEX MATCH "^([0-9]+)" _ignored "${CMAKE_CXX_COMPILER_VERSION}")
        set(CONAN_SETTINGS
            "compiler=clang"
            "compiler.version=${CMAKE_MATCH_1}"
            "compiler.libcxx=libstdc++11"
        )
    endif()
    # Qualcomm QCS5430 AI Box runs AArch64 — request the ARM64 Conan package
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        list(APPEND CONAN_SETTINGS "arch=armv8")
    endif()
endif()

set(OPENCV_OPTIONS
    opencv:parallel=False
    opencv:contrib=True
    opencv:contrib_freetype=False
    opencv:contrib_sfm=False
    opencv:with_jpeg=libjpeg
    opencv:with_png=False
    opencv:with_tiff=False
    opencv:with_jpeg2000=False
    opencv:with_openexr=False
    opencv:with_eigen=False
    opencv:with_webp=False
    opencv:with_quirc=False
    opencv:with_cuda=False
    opencv:with_cublas=False
    opencv:with_ffmpeg=False
)

if(UNIX)
    set(OPENCV_OPTIONS 
        ${OPENCV_OPTIONS}
        opencv:fPIC=True
        opencv:with_gtk=False
        opencv:with_cufft=False
        opencv:with_v4l=False
    )
endif()

conan_cmake_run(BUILD_TYPE "Release" SETTINGS ${CONAN_SETTINGS})

conan_cmake_configure(REQUIRES opencv/4.5.5
    GENERATORS cmake_find_package
    OPTIONS ${OPENCV_OPTIONS}
    SETTINGS ${CONAN_SETTINGS}
)

conan_cmake_install(PATH_OR_REFERENCE .
    OPTIONS ${OPENCV_OPTIONS}
    SETTINGS ${CONAN_SETTINGS}
    BUILD missing
)


find_package(OpenCV REQUIRED)

# The plugin no longer downloads or loads local model assets. Inference runs through the
# colocated Python analytics service, so the build only needs OpenCV utility modules.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(UNIX)
    # In Linux, for the plugin .so library, set `rpath` to "$ORIGIN" and do not set `runpath`, thus
    # enabling the lookup of the dependencies in the plugin dir first.
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,--disable-new-dtags")
endif()
set(CMAKE_SKIP_BUILD_RPATH ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN")

if(WIN32)
    set(API_EXPORT_MACRO "__declspec(dllexport)")
elseif(UNIX)
    set(API_EXPORT_MACRO "__attribute__((visibility(\"default\")))")
endif()

#--------------------------------------------------------------------------------------------------
# Define nx_kit lib, static.

set(nxKitLibraryType "STATIC" CACHE STRING "" FORCE)
set(nxKitWithTests "YES" CACHE STRING "" FORCE)

add_subdirectory(${metadataSdkDir}/nx_kit ${CMAKE_CURRENT_BINARY_DIR}/nx_kit)

#--------------------------------------------------------------------------------------------------
# Define nx_sdk lib, static, depends on nx_kit.

set(SDK_SRC_DIR ${metadataSdkDir}/src)
file(GLOB_RECURSE SDK_SRC ${SDK_SRC_DIR}/*)

add_library(nx_sdk STATIC ${SDK_SRC})
target_include_directories(nx_sdk PUBLIC ${SDK_SRC_DIR})
target_link_libraries(nx_sdk PRIVATE nx_kit)

target_compile_definitions(nx_sdk PRIVATE NX_PLUGIN_API=${API_EXPORT_MACRO})

#--------------------------------------------------------------------------------------------------
# Define opencv_object_detection_analytics_plugin lib, dynamic, depends on nx_kit and nx_sdk.

set(OPENCV_OBJECT_DETECTION_ANALYTICS_PLUGIN_SRC_DIR
    ${PROJECT_ROOT}/src/sample_company/vms_server_plugins/opencv_object_detection)
file(GLOB_RECURSE OPENCV_OBJECT_DETECTION_ANALYTICS_PLUGIN_SRC
    ${OPENCV_OBJECT_DETECTION_ANALYTICS_PLUGIN_SRC_DIR}/*)

add_library(yolo26_people_analytics_plugin SHARED
    ${OPENCV_OBJECT_DETECTION_ANALYTICS_PLUGIN_SRC})
# /Od (MSVC) disables optimizations on the tracker to ease debugging; GCC/Clang
# use -O0 automatically in Debug builds so no equivalent flag is needed there.
if(MSVC)
    set_source_files_properties(
        ${OPENCV_OBJECT_DETECTION_ANALYTICS_PLUGIN_SRC_DIR}/object_tracker.cpp
        PROPERTIES COMPILE_FLAGS "/Od"
    )
endif()
target_include_directories(yolo26_people_analytics_plugin PRIVATE
    ${OPENCV_OBJECT_DETECTION_ANALYTICS_PLUGIN_SRC_DIR}
    ${PROJECT_ROOT}/3rd_party
    ${CMAKE_BINARY_DIR}/include   # plugin_version.h (generated)
)
target_link_libraries(yolo26_people_analytics_plugin
    nx_kit
    nx_sdk
    opencv::core opencv::flann opencv::imgproc opencv::imgcodecs opencv::ml 
    opencv::plot opencv::opencv_features2d opencv::opencv_calib3d opencv::datasets opencv::video opencv::tracking
)

if (WIN32)
    target_link_libraries(yolo26_people_analytics_plugin
        ws2_32
    )
endif()

target_compile_definitions(yolo26_people_analytics_plugin
    PRIVATE NX_PLUGIN_API=${API_EXPORT_MACRO}
)

# ── Release artifact packaging ────────────────────────────────────────────────
# Run:  cmake --install <buildDir> --prefix <destDir>
# Produces a flat directory ready for deployment or zipping:
#   <destDir>/yolo26_people_analytics_plugin.dll   (Windows)
#   <destDir>/yolo26_people_analytics_plugin.so    (Linux)
#   <destDir>/manifest.json
#   <destDir>/version_info.json
install(TARGETS yolo26_people_analytics_plugin
    LIBRARY DESTINATION .    # .so on Linux
    RUNTIME DESTINATION .    # .dll on Windows
)
install(FILES
    "${CMAKE_CURRENT_LIST_DIR}/manifest.json"
    "${CMAKE_BINARY_DIR}/version_info.json"
    DESTINATION .
)
