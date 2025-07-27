cmake_minimum_required(VERSION 3.16)

# ============================================================================
# Project Configuration
# ============================================================================
project(Blackchirp 
    VERSION 1.1.0
    DESCRIPTION "CP-FTMW Spectroscopy Software"
    LANGUAGES CXX
)

# Set release version
set(BC_RELEASE_VERSION "devel" CACHE STRING "Release version string")

# Get git build version
execute_process(
    COMMAND git describe --always
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE BC_BUILD_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(NOT BC_BUILD_VERSION)
    set(BC_BUILD_VERSION "unknown")
endif()

# Version definitions for compatibility with qmake system
set(BC_MAJOR_VERSION ${PROJECT_VERSION_MAJOR})
set(BC_MINOR_VERSION ${PROJECT_VERSION_MINOR})
set(BC_PATCH_VERSION ${PROJECT_VERSION_PATCH})

# Viewer-specific versioning
set(BCV_MAJOR_VERSION 1)
set(BCV_MINOR_VERSION 0)
set(BCV_PATCH_VERSION 0)
set(BCV_RELEASE_VERSION "alpha")

# ============================================================================
# Build Options and Configuration
# ============================================================================

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

# Global options
option(BC_BUILD_VIEWER_ONLY "Build only the viewer application" OFF)
option(BC_BUILD_TESTS "Build test executables" ON)
option(BC_BUILD_DOCUMENTATION "Build documentation" OFF)

# Hardware and build configuration is now handled by dedicated config files

# ============================================================================
# Compiler Configuration
# ============================================================================

# C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/W4)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2)
    endif()
endif()

# ============================================================================
# Qt Configuration
# ============================================================================

# Enable Qt's CMake integration
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Find Qt6 components
set(QT_REQUIRED_COMPONENTS Core Gui Widgets Network)

# SerialPort only needed for main application, not viewer
if(NOT BC_BUILD_VIEWER_ONLY)
    list(APPEND QT_REQUIRED_COMPONENTS SerialPort)
endif()

find_package(Qt6 REQUIRED COMPONENTS ${QT_REQUIRED_COMPONENTS})

# Qt version information
message(STATUS "Using Qt ${Qt6_VERSION}")

# ============================================================================
# External Dependencies
# ============================================================================

# Add cmake directory to module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# QWT - Scientific plotting library
find_package(QWT REQUIRED)

# GSL - GNU Scientific Library
find_package(GSL REQUIRED)

# CUDA - Optional for GPU acceleration
if(BC_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    # CUDA compilation flags
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Architecture-specific flags
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 50 60 70 75 80 86)
    endif()
    
    message(STATUS "CUDA enabled with architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Platform-specific libraries
if(WIN32)
    # Windows-specific dependencies
    find_library(WINMM_LIBRARY winmm)
elseif(APPLE)
    find_library(COCOA_LIBRARY Cocoa)
    find_library(IOKIT_LIBRARY IOKit)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
else()
    # Linux-specific dependencies
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(UDEV libudev)
    endif()
endif()

# ============================================================================
# Include Directories
# ============================================================================

# Global include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# ============================================================================
# Hardware Configuration Processing
# ============================================================================

# Check for user build configuration file
set(BUILD_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildConfig.cmake")
set(BUILD_CONFIG_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildConfig.cmake.template")

if(NOT EXISTS ${BUILD_CONFIG_FILE})
    message(STATUS "")
    message(STATUS "=== BUILD CONFIGURATION SETUP ===")
    message(STATUS "")
    message(STATUS "No build configuration file found. Creating default configuration...")
    message(STATUS "")
    message(STATUS "The following file has been created with default build settings:")
    message(STATUS "  ${BUILD_CONFIG_FILE}")
    message(STATUS "")
    message(STATUS "You can edit this file to customize build options, enable optional")
    message(STATUS "modules (LIF, CUDA), or change development settings. Most users")
    message(STATUS "can use the default settings without modification.")
    message(STATUS "")
    
    # Copy template to create user config file
    configure_file(${BUILD_CONFIG_TEMPLATE} ${BUILD_CONFIG_FILE} COPYONLY)
endif()

# Include user build configuration
include(${BUILD_CONFIG_FILE})

# Check for user hardware configuration file
set(HARDWARE_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HardwareConfig.cmake")
set(HARDWARE_CONFIG_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HardwareConfig.cmake.template")

if(NOT EXISTS ${HARDWARE_CONFIG_FILE})
    message(STATUS "")
    message(STATUS "=== HARDWARE CONFIGURATION REQUIRED ===")
    message(STATUS "")
    message(STATUS "No hardware configuration file found. Creating default configuration...")
    message(STATUS "")
    message(STATUS "IMPORTANT: Please edit the following file to configure your hardware:")
    message(STATUS "  ${HARDWARE_CONFIG_FILE}")
    message(STATUS "")
    message(STATUS "The file contains detailed instructions and lists all available")
    message(STATUS "hardware implementations. Default 'virtual' implementations are")
    message(STATUS "provided for testing, but you should configure real hardware")
    message(STATUS "for actual experiments.")
    message(STATUS "")
    message(STATUS "After editing the configuration, re-run cmake to apply changes.")
    message(STATUS "")
    
    # Copy template to create user config file
    configure_file(${HARDWARE_CONFIG_TEMPLATE} ${HARDWARE_CONFIG_FILE} COPYONLY)
endif()

# Include user hardware configuration
include(${HARDWARE_CONFIG_FILE})

# Include hardware validation
include(cmake/HardwareValidation.cmake)

# Process clock configuration
string(REPLACE ";" " " BC_CLOCKS_SPACE_SEPARATED "${BC_CLOCKS}")
list(LENGTH BC_CLOCKS BC_NUM_CLOCKS)

# Validate hardware configuration
validate_hardware_configuration()

# ============================================================================
# Version and Configuration Definitions
# ============================================================================

# Create a function to add version definitions to targets
function(add_blackchirp_definitions target_name)
    target_compile_definitions(${target_name} PRIVATE
        BC_MAJOR_VERSION=${BC_MAJOR_VERSION}
        BC_MINOR_VERSION=${BC_MINOR_VERSION}
        BC_PATCH_VERSION=${BC_PATCH_VERSION}
        BC_RELEASE_VERSION=${BC_RELEASE_VERSION}
        BC_BUILD_VERSION="${BC_BUILD_VERSION}"
        BC_NUM_CLOCKS=${BC_NUM_CLOCKS}
    )
    
    # Hardware-specific definitions are now handled by BlackchirpHardware.cmake module
    
    # All hardware definitions are now handled by BlackchirpHardware.cmake module
    
    # LIF definitions
    if(BC_ENABLE_LIF)
        target_compile_definitions(${target_name} PRIVATE BC_LIF)
    endif()
    
    # CUDA definitions
    if(BC_ENABLE_CUDA)
        target_compile_definitions(${target_name} PRIVATE BC_CUDA)
    endif()
endfunction()

# Viewer-specific version function
function(add_viewer_definitions target_name)
    target_compile_definitions(${target_name} PRIVATE
        BC_VIEWER
        BCV_MAJOR_VERSION=${BCV_MAJOR_VERSION}
        BCV_MINOR_VERSION=${BCV_MINOR_VERSION}
        BCV_PATCH_VERSION=${BCV_PATCH_VERSION}
        BCV_RELEASE_VERSION=${BCV_RELEASE_VERSION}
        BCV_BUILD_VERSION="${BC_BUILD_VERSION}"
    )
endfunction()

# ============================================================================
# Data Layer
# ============================================================================

# Create data layer library (shared between main app and viewer)
include(cmake/BlackchirpData.cmake)

# ============================================================================
# GUI Layer
# ============================================================================

# Create GUI layer library (includes widgets, plots, and models)
if(BC_BUILD_VIEWER_ONLY)
    # Viewer-only build: use lightweight GUI without hardware dependencies
    include(cmake/BlackchirpViewerGui.cmake)
else()
    # Full application build: use complete GUI layer
    include(cmake/BlackchirpGui.cmake)
    
    # Also create viewer GUI library for viewer application
    include(cmake/BlackchirpViewerGui.cmake)
endif()

# ============================================================================
# Hardware Layer
# ============================================================================

# Create hardware layer library (only for main application, not viewer)
if(NOT BC_BUILD_VIEWER_ONLY)
    include(cmake/BlackchirpHardware.cmake)
endif()

# ============================================================================
# Application Layer
# ============================================================================

# Create main application executable (only for main application, not viewer)
if(NOT BC_BUILD_VIEWER_ONLY)
    include(cmake/BlackchirpApplication.cmake)
endif()

# ============================================================================
# Subdirectories
# ============================================================================

# Include CMake modules
add_subdirectory(cmake)

# Always build tests (they have their own CMakeLists.txt)
if(BC_BUILD_TESTS)
    enable_testing()
    # Note: Test CMakeLists.txt is already in root, will be updated separately
endif()

# Build main application unless viewer-only build
# Note: Main application is now built via BlackchirpApplication.cmake module
# The src/CMakeLists.txt is kept for reference but not used

# Build viewer application
include(cmake/BlackchirpViewerApplication.cmake)

# Documentation
if(BC_BUILD_DOCUMENTATION)
    include(cmake/BlackchirpDocumentation.cmake)
endif()

# ============================================================================
# Installation and Packaging
# ============================================================================

# Install configuration
include(GNUInstallDirs)

# CPack configuration
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include(cmake/Packaging.cmake)
endif()

# ============================================================================
# Information Summary
# ============================================================================

message(STATUS "")
message(STATUS "Blackchirp ${PROJECT_VERSION} Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Qt version: ${Qt6_VERSION}")
message(STATUS "  Build viewer only: ${BC_BUILD_VIEWER_ONLY}")
message(STATUS "  Enable tests: ${BC_BUILD_TESTS}")
message(STATUS "  Enable LIF: ${BC_ENABLE_LIF}")
message(STATUS "  Enable CUDA: ${BC_ENABLE_CUDA}")
message(STATUS "")
