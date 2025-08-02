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

# Always build tests
if(BC_BUILD_TESTS)
    enable_testing()
    
    # Create test executables
    find_package(Qt6 REQUIRED COMPONENTS Test)
    
    # Settings Storage Test
    add_executable(tst_settingsstoragetest tests/tst_settingsstoragetest.cpp)
    target_link_libraries(tst_settingsstoragetest 
        blackchirp-data 
        Qt6::Test 
        Qt6::Core
    )
    add_blackchirp_definitions(tst_settingsstoragetest)
    add_test(NAME SettingsStorageTest COMMAND tst_settingsstoragetest)
    
    # Header Storage Test  
    add_executable(tst_headerstoragetest tests/tst_headerstoragetest.cpp)
    target_link_libraries(tst_headerstoragetest 
        blackchirp-data 
        Qt6::Test 
        Qt6::Core
    )
    add_blackchirp_definitions(tst_headerstoragetest)
    add_test(NAME HeaderStorageTest COMMAND tst_headerstoragetest)
    
    # BlackChirp CSV Test
    add_executable(tst_blackchirpcsvtest tests/tst_blackchirpcsv.cpp)
    target_link_libraries(tst_blackchirpcsvtest 
        blackchirp-data 
        Qt6::Test 
        Qt6::Core
    )
    add_blackchirp_definitions(tst_blackchirpcsvtest)
    add_test(NAME BlackChirpCSVTest COMMAND tst_blackchirpcsvtest)
    
    # FT Worker Test
    add_executable(tst_ftworkertest tests/tst_ftworkertest.cpp)
    target_link_libraries(tst_ftworkertest 
        blackchirp-data 
        Qt6::Test 
        Qt6::Core
        GSL::gsl
    )
    add_blackchirp_definitions(tst_ftworkertest)
    add_test(NAME FTWorkerTest COMMAND tst_ftworkertest)
    
    # Background Convolution Test - temporarily disabled due to API updates needed
    # add_executable(tst_background_convolution tests/tst_background_convolution.cpp)
    # target_link_libraries(tst_background_convolution 
    #     blackchirp-data 
    #     Qt6::Test 
    #     Qt6::Core
    #     GSL::gsl
    # )
    # add_blackchirp_definitions(tst_background_convolution)
    # add_test(NAME BackgroundConvolutionTest COMMAND tst_background_convolution)
    
    # Generic XY Parser Test
    add_executable(tst_genericxyparser tests/tst_genericxyparser.cpp)
    target_link_libraries(tst_genericxyparser 
        blackchirp-data 
        Qt6::Test 
        Qt6::Core
    )
    add_blackchirp_definitions(tst_genericxyparser)
    target_compile_definitions(tst_genericxyparser PRIVATE 
        TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/testdata"
    )
    add_test(NAME GenericXYParserTest COMMAND tst_genericxyparser)
    
    # Operation Capability Test
    add_executable(tst_operation_capability_only tests/tst_operation_capability_only.cpp)
    target_link_libraries(tst_operation_capability_only 
        blackchirp-data 
        blackchirp-gui
        Qt6::Test 
        Qt6::Core
        Qt6::Widgets
    )
    add_blackchirp_definitions(tst_operation_capability_only)
    add_test(NAME OperationCapabilityTest COMMAND tst_operation_capability_only)
    
    # Overlay Operations Simple Test
    add_executable(tst_overlayoperations_simple tests/tst_overlayoperations_simple.cpp)
    target_link_libraries(tst_overlayoperations_simple 
        blackchirp-data 
        blackchirp-gui
        Qt6::Test 
        Qt6::Core
        Qt6::Widgets
    )
    add_blackchirp_definitions(tst_overlayoperations_simple)
    add_test(NAME OverlayOperationsSimpleTest COMMAND tst_overlayoperations_simple)
    
    # SPCAT Parser Test
    add_executable(tst_spcatparser tests/tst_spcatparser.cpp)
    target_link_libraries(tst_spcatparser 
        blackchirp-data 
        Qt6::Test 
        Qt6::Core
    )
    add_blackchirp_definitions(tst_spcatparser)
    target_compile_definitions(tst_spcatparser PRIVATE 
        TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/testdata"
    )
    add_test(NAME SPCATParserTest COMMAND tst_spcatparser)
    
    # XIAM Parser Test
    add_executable(tst_xiamparser tests/tst_xiamparser.cpp)
    target_link_libraries(tst_xiamparser 
        blackchirp-data 
        Qt6::Test 
        Qt6::Core
    )
    add_blackchirp_definitions(tst_xiamparser)
    add_test(NAME XIAMParserTest COMMAND tst_xiamparser)
    
    # Hardware Registry Comprehensive Test
    add_executable(tst_hardwareregistrytest tests/tst_hardwareregistrytest.cpp)
    target_link_libraries(tst_hardwareregistrytest 
        blackchirp-hardware 
        Qt6::Test 
        Qt6::Core
    )
    add_blackchirp_definitions(tst_hardwareregistrytest)
    add_test(NAME HardwareRegistryTest COMMAND tst_hardwareregistrytest)
    
    # Runtime Hardware Config Comprehensive Test
    add_executable(tst_runtimehardwareconfigtest tests/tst_runtimehardwareconfigtest.cpp)
    target_link_libraries(tst_runtimehardwareconfigtest 
        blackchirp-hardware 
        Qt6::Test 
        Qt6::Core
    )
    add_blackchirp_definitions(tst_runtimehardwareconfigtest)
    add_test(NAME RuntimeHardwareConfigTest COMMAND tst_runtimehardwareconfigtest)
    
    # Hardware Profile Manager Test
    add_executable(tst_hardwareprofilemanagertest tests/tst_hardwareprofilemanagertest.cpp)
    target_link_libraries(tst_hardwareprofilemanagertest 
        blackchirp-hardware 
        Qt6::Test 
        Qt6::Core
    )
    add_blackchirp_definitions(tst_hardwareprofilemanagertest)
    add_test(NAME HardwareProfileManagerTest COMMAND tst_hardwareprofilemanagertest)
    
    # Test Hardware Library - includes core hardware sources plus all virtual implementations
    # This replaces the main blackchirp-hardware library for testing to avoid symbol conflicts
    add_library(blackchirp-test-hardware STATIC
        # Core hardware sources (from BlackchirpHardware.cmake) - excluding hardwaremanager
        src/hardware/core/communication/communicationprotocol.cpp
        src/hardware/core/communication/rs232instrument.cpp
        src/hardware/core/communication/tcpinstrument.cpp
        src/hardware/core/communication/virtualinstrument.cpp
        src/hardware/core/communication/gpibinstrument.cpp
        src/hardware/core/communication/custominstrument.cpp
        src/hardware/core/hardwareobject.cpp
        src/hardware/core/clock/clock.cpp
        src/hardware/core/ftmwdigitizer/ftmwscope.cpp
        src/hardware/optional/chirpsource/awg.cpp
        src/hardware/optional/pulsegenerator/pulsegenerator.cpp
        src/hardware/optional/flowcontroller/flowcontroller.cpp
        src/hardware/optional/gpibcontroller/gpibcontroller.cpp
        src/hardware/optional/ioboard/ioboard.cpp
        src/hardware/optional/pressurecontroller/pressurecontroller.cpp
        src/hardware/optional/tempcontroller/temperaturecontroller.cpp
        
        # Virtual hardware implementations for comprehensive testing
        src/hardware/optional/gpibcontroller/virtualgpibcontroller.cpp
        src/hardware/optional/chirpsource/virtualawg.cpp
        src/hardware/optional/pulsegenerator/virtualpulsegenerator.cpp
        src/hardware/optional/flowcontroller/virtualflowcontroller.cpp
        src/hardware/optional/ioboard/virtualioboard.cpp
        src/hardware/optional/pressurecontroller/virtualpressurecontroller.cpp
        src/hardware/optional/tempcontroller/virtualtempcontroller.cpp
        src/hardware/core/ftmwdigitizer/virtualftmwscope.cpp
        src/hardware/core/clock/fixedclock.cpp
    )
    
    target_link_libraries(blackchirp-test-hardware
        PUBLIC blackchirp-data
        PUBLIC Qt6::Core
        PUBLIC Qt6::SerialPort
        PUBLIC Qt6::Network
    )
    
    target_include_directories(blackchirp-test-hardware
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    
    add_blackchirp_definitions(blackchirp-test-hardware)
    
    # Enable GPIB controller for comprehensive testing
    target_compile_definitions(blackchirp-test-hardware PRIVATE BC_GPIBCONTROLLER)

    # Hardware Communication Test
    # TODO: Temporarily disabled due to HardwareManager dependencies in HardwareObject
    # This test will be re-enabled once Phase 2 runtime hardware migration is complete
    # and HardwareManager is updated to work with the new runtime configuration system.
    # The test itself is valuable for validating GPIB modernization improvements.
    #
    # add_executable(tst_hardwarecommunication tests/tst_hardwarecommunication.cpp)
    # target_link_libraries(tst_hardwarecommunication 
    #     blackchirp-test-hardware  # Only link test hardware library, not main one
    #     blackchirp-data 
    #     Qt6::Test 
    #     Qt6::Core
    # )
    # add_blackchirp_definitions(tst_hardwarecommunication)
    # add_test(NAME HardwareCommunicationTest COMMAND tst_hardwarecommunication)
    
    # Create a custom target to build all tests
    add_custom_target(tests DEPENDS
        tst_settingsstoragetest
        tst_headerstoragetest
        tst_blackchirpcsvtest
        tst_ftworkertest
        tst_genericxyparser
        tst_operation_capability_only
        tst_overlayoperations_simple
        tst_spcatparser
        tst_xiamparser
        tst_hardwareregistrytest
        tst_runtimehardwareconfigtest
        # tst_hardwarecommunication  # Disabled until HardwareManager migration complete
    )
    
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
