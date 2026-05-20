cmake_minimum_required(VERSION 3.16)

# ============================================================================
# Project Configuration
# ============================================================================
project(Blackchirp
    VERSION 2.0.0
    DESCRIPTION "CP-FTMW Spectroscopy Software"
    LANGUAGES CXX
)

# Release-stage tag baked into the version string and on-disk schema.
# Plain set(): always reflects the source-tree value at configure time.
# CACHE STRING was tempting (overridable from `-D...` at configure) but
# silently sticky across releases — a checkout that switched from
# `release` back to `alpha` would still emit packages stamped `release`
# until someone manually cleared the cache.
set(BC_RELEASE_VERSION "alpha2")

# Get git build version
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE BC_BUILD_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(NOT BC_BUILD_VERSION)
    set(BC_BUILD_VERSION "unknown")
endif()

# Version definitions. The numeric version has a single source of
# truth — the project() VERSION above — and the release-stage tag has
# a single source of truth — BC_RELEASE_VERSION above. Everything
# else, including the viewer-specific (BCV_*) version, is derived so a
# release bump touches exactly those two lines (and conf.py, the CI
# workflow, and the on-disk schema all read back from project()).
set(BC_MAJOR_VERSION ${PROJECT_VERSION_MAJOR})
set(BC_MINOR_VERSION ${PROJECT_VERSION_MINOR})
set(BC_PATCH_VERSION ${PROJECT_VERSION_PATCH})

# The viewer ships from this same tree and is versioned in lockstep
# with the acquisition app; alias rather than duplicate.
set(BCV_MAJOR_VERSION ${PROJECT_VERSION_MAJOR})
set(BCV_MINOR_VERSION ${PROJECT_VERSION_MINOR})
set(BCV_PATCH_VERSION ${PROJECT_VERSION_PATCH})
set(BCV_RELEASE_VERSION ${BC_RELEASE_VERSION})

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

# ============================================================================
# Compiler Configuration
# ============================================================================

# C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific flags. Release builds keep debug info (-g / /Zi)
# so the crash-handler stacktraces can be resolved with addr2line / atos
# / WinDbg against companion .debug / .dSYM / .pdb files captured in
# CI. Stripping happens at install time in cmake/Packaging.cmake.
if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    add_compile_options(-Wall -Wextra)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -g)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/W4)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2 /Zi)
        add_link_options(/DEBUG)
    endif()
endif()

# ============================================================================
# Build Configuration (loaded early so local hooks can affect package discovery)
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

# Include user build configuration. Loaded before find_package() so local
# customizations (CMAKE_PREFIX_PATH tweaks, conda shielding, vendor SDK
# paths, etc.) can influence dependency discovery.
include(${BUILD_CONFIG_FILE})

# ============================================================================
# Qt Configuration
# ============================================================================

# Enable Qt's CMake integration
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Find Qt6 components
set(QT_REQUIRED_COMPONENTS Core Gui Widgets Network Concurrent)

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

# Eigen3 - Header-only linear algebra (used by Analysis/PeakFinder)
find_package(Eigen3 REQUIRED NO_MODULE)

# Standard install-directory variables (CMAKE_INSTALL_BINDIR, _DATADIR, etc.)
# Must be available before any subdirectory install() rules use them.
include(GNUInstallDirs)

# Qt redistributable bundling for Windows/macOS (no-op on Linux). Provides
# blackchirp_deploy_qt(<target>) used by the app cmake modules.
include(cmake/QtDeployment.cmake)

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
        BC_QWT_VERSION="${QWT_VERSION}"
        BC_GSL_VERSION="${GSL_VERSION}"
        BC_EIGEN3_VERSION="${Eigen3_VERSION}"
    )
    
    # CUDA is compile-time; all hardware selection is runtime-only
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

    # ZoomPanPlot Thread Safety Test
    # blackchirp-gui has unresolved references into application sources that
    # the main blackchirp executable provides directly (AcquisitionManager,
    # BatchManager, mainwindow). Rather than dragging those in, the test
    # compiles only the plot-layer translation units it needs.
    add_executable(tst_zoompanplotthreadsafety
        tests/tst_zoompanplotthreadsafety.cpp
        src/gui/plot/zoompanplot.cpp
        src/gui/plot/ftplot.cpp
        src/gui/plot/blackchirpplotcurve.cpp
        src/gui/plot/curvefactory.cpp
        src/gui/plot/customtracker.cpp
        src/gui/plot/customzoomer.cpp
        src/gui/plot/curveappearancewidget.cpp
        src/gui/plot/curveappearancepresetmanager.cpp
        src/gui/plot/presetsavedialog.cpp
        src/gui/widget/settingstable.cpp
        src/gui/style/themecolors.cpp
        src/gui/util/numericformat.cpp
    )
    target_include_directories(tst_zoompanplotthreadsafety PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(tst_zoompanplotthreadsafety
        blackchirp-data
        Qt6::Test
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        Qt6::Concurrent
        QWT::QWT
    )
    add_blackchirp_definitions(tst_zoompanplotthreadsafety)
    add_test(NAME ZoomPanPlotThreadSafetyTest COMMAND tst_zoompanplotthreadsafety)
    set_tests_properties(ZoomPanPlotThreadSafetyTest PROPERTIES
        ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
    )

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
    
    # Hardware Keys Test
    add_executable(tst_hardwarekeys tests/tst_hardwarekeys.cpp)
    target_link_libraries(tst_hardwarekeys
        blackchirp-data
        Qt6::Test
        Qt6::Core
    )
    add_blackchirp_definitions(tst_hardwarekeys)
    add_test(NAME HardwareKeysTest COMMAND tst_hardwarekeys)

    # Experiment Loading Test
    add_executable(tst_experimentloading tests/tst_experimentloading.cpp)
    target_link_libraries(tst_experimentloading
        blackchirp-test-hardware
        blackchirp-data
        blackchirp-hardware
        Qt6::Test
        Qt6::Core
    )
    add_blackchirp_definitions(tst_experimentloading)
    target_compile_definitions(tst_experimentloading PRIVATE
        TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/testdata"
    )
    add_test(NAME ExperimentLoadingTest COMMAND tst_experimentloading)

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
        src/hardware/core/ftmwdigitizer/ftmwdigitizer.cpp
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
        src/hardware/core/ftmwdigitizer/virtualftmwdigitizer.cpp
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
    
    # No hardware-specific compile definitions needed - all hardware available at runtime

    # ScientificSpinBox Test — compile only the required sources to avoid pulling
    # in the full blackchirp-gui library (which has hardware/singleton dependencies)
    add_executable(tst_scientificspinboxtest
        tests/tst_scientificspinboxtest.cpp
        src/gui/widget/scientificspinbox.cpp
        src/gui/util/numericformat.cpp
        src/gui/style/themecolors.cpp
    )
    target_include_directories(tst_scientificspinboxtest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(tst_scientificspinboxtest
        Qt6::Test
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
    )
    add_blackchirp_definitions(tst_scientificspinboxtest)
    add_test(NAME ScientificSpinBoxTest COMMAND tst_scientificspinboxtest)
    set_tests_properties(ScientificSpinBoxTest PROPERTIES
        ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
    )

    # Waveform Buffer Test
    add_executable(tst_waveformbuffertest tests/tst_waveformbuffertest.cpp)
    target_link_libraries(tst_waveformbuffertest
        blackchirp-data
        Qt6::Test
        Qt6::Core
    )
    add_blackchirp_definitions(tst_waveformbuffertest)
    add_test(NAME WaveformBufferTest COMMAND tst_waveformbuffertest)

    add_executable(tst_loadoutmanagertest tests/tst_loadoutmanagertest.cpp)
    target_link_libraries(tst_loadoutmanagertest
        blackchirp-data
        Qt6::Test
        Qt6::Core
    )
    add_blackchirp_definitions(tst_loadoutmanagertest)
    add_test(NAME LoadoutManagerTest COMMAND tst_loadoutmanagertest)

    # Hardware Communication Test
    # Disabled: HardwareObject carries HardwareManager dependencies that conflict
    # with the test hardware library; re-enable once that coupling is resolved.
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
        tst_zoompanplotthreadsafety
        tst_spcatparser
        tst_xiamparser
        tst_hardwareregistrytest
        tst_runtimehardwareconfigtest
        tst_hardwareprofilemanagertest
        tst_hardwarekeys
        tst_experimentloading
        tst_scientificspinboxtest
        tst_waveformbuffertest
        tst_loadoutmanagertest
        # tst_hardwarecommunication  # Disabled (see comment above)
    )
    
endif()

# Build main application unless viewer-only build

# Build viewer application
include(cmake/BlackchirpViewerApplication.cmake)

# Documentation
if(BC_BUILD_DOCUMENTATION)
    include(cmake/BlackchirpDocumentation.cmake)
endif()

# ============================================================================
# Installation and Packaging
# ============================================================================

# Install configuration (GNUInstallDirs is included earlier, near the
# top of the dependency block, so subdirectory install() rules can use
# CMAKE_INSTALL_BINDIR/_DATADIR/etc. at registration time).

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
message(STATUS "  Enable CUDA: ${BC_ENABLE_CUDA}")
message(STATUS "")
