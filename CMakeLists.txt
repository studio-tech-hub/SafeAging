cmake_minimum_required(VERSION 3.16)
project(yolov8_flow2_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(NX_METADATA_SDK_DIR "" CACHE PATH "Path to unpacked Nx Metadata SDK directory")
if(NX_METADATA_SDK_DIR STREQUAL "")
    message(FATAL_ERROR "NX_METADATA_SDK_DIR is required")
endif()

if(NOT EXISTS "${NX_METADATA_SDK_DIR}/src")
    message(FATAL_ERROR "NX_METADATA_SDK_DIR does not contain src/")
endif()

if(NOT EXISTS "${NX_METADATA_SDK_DIR}/nx_kit")
    message(FATAL_ERROR "NX_METADATA_SDK_DIR does not contain nx_kit/")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(UNIX)
    set(API_EXPORT_MACRO "__attribute__((visibility(\"default\")))")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,--disable-new-dtags")
    set(CMAKE_SKIP_BUILD_RPATH ON)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
elseif(WIN32)
    set(API_EXPORT_MACRO "__declspec(dllexport)")
endif()

set(nxKitLibraryType "STATIC" CACHE STRING "" FORCE)
set(nxKitWithTests "NO" CACHE STRING "" FORCE)
add_subdirectory("${NX_METADATA_SDK_DIR}/nx_kit" "${CMAKE_BINARY_DIR}/nx_kit")

file(GLOB_RECURSE SDK_SRC CONFIGURE_DEPENDS "${NX_METADATA_SDK_DIR}/src/*")
add_library(nx_sdk STATIC ${SDK_SRC})
target_include_directories(nx_sdk PUBLIC "${NX_METADATA_SDK_DIR}/src")
target_link_libraries(nx_sdk PRIVATE nx_kit)
target_compile_definitions(nx_sdk PRIVATE NX_PLUGIN_API=${API_EXPORT_MACRO})

add_library(yolov8_flow2_plugin SHARED
    src/plugin.cpp
    src/engine.cpp
    src/device_agent.cpp
    src/object_detector.cpp
    src/utils/base64.cpp
    src/utils/image_codec.cpp
)

target_include_directories(yolov8_flow2_plugin PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rd_party"
    "${NX_METADATA_SDK_DIR}/src"
)

target_link_libraries(yolov8_flow2_plugin PRIVATE
    nx_kit
    nx_sdk
    ${OpenCV_LIBS}
)

target_compile_definitions(yolov8_flow2_plugin PRIVATE NX_PLUGIN_API=${API_EXPORT_MACRO})

