# BlackchirpData.cmake - Data layer components for Blackchirp
#
# This module defines the blackchirp-data library target containing:
# - Analysis and signal processing components
# - Experiment configuration and management 
# - Data storage and CSV handling
# - File parsers and data models
# - Overlay processing system

# Include guard to prevent multiple inclusions
if(BLACKCHIRP_DATA_CMAKE_INCLUDED)
    return()
endif()
set(BLACKCHIRP_DATA_CMAKE_INCLUDED TRUE)

# ============================================================================
# Data Layer Source Files
# ============================================================================

set(BLACKCHIRP_DATA_SOURCES
    # Core data components
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/loghandler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/bcglobals.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/crashhandler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/updatechecker.cpp
    
    # Analysis and signal processing
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/analysis/analysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/analysis/ft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/analysis/ftworker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/analysis/peakfinder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/analysis/waveformparser.cpp
    
    # Experiment configuration
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/chirpconfig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/digitizerconfig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/experiment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/hardwaredatacontainer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/experimentobjective.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/experimentvalidator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/fid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/ftmwconfig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/ftmwconfigtypes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/overlaybase.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/overlaytypes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/catalogdata.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/rfconfig.cpp

    # Loadout system
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/loadout/rfconfigsnapshot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/loadout/chirpconfigloadout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/loadout/ftmwdigitizerloadout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/loadout/hardwareloadout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/loadout/loadoutmanager.cpp

    # Hardware configuration classes (moved from hardware layer - pure data structures)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/hardware/core/ftmwdigitizerconfig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/hardware/optional/pulsegenerator/pulsegenconfig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/hardware/optional/flowcontroller/flowconfig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/hardware/optional/ioboard/ioboardconfig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/hardware/optional/pressurecontroller/pressurecontrollerconfig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/hardware/optional/tempcontroller/temperaturecontrollerconfig.cpp
    
    # File parsers
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/processing/parsers/fileparser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/processing/parsers/catalogparser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/processing/parsers/fileparserregistry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/processing/parsers/spcatparser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/processing/parsers/xiamparser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/processing/parsers/genericxyparser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/processing/parsers/genericxydata.cpp
    
    # Data models moved to GUI layer - they depend on Qt widgets
    
    # Processing system
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/processing/overlayoperation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/processing/overlayprocessmanager.cpp

    # Storage system
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/applicationconfigmanager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/auxdatastorage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/blackchirpcsv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/datastoragebase.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/fidmultistorage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/fidpeakupstorage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/fidsinglestorage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/fidstoragebase.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/headerstorage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/overlaystorage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/settingsstorage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/waveformbuffer.cpp
)

set(BLACKCHIRP_DATA_HEADERS
    # Core data components
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/loghandler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/bcglobals.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/crashhandler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/crashhandler_p.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/updatechecker.h
    
    # Analysis and signal processing
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/analysis/analysis.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/analysis/ft.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/analysis/ftworker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/analysis/peakfinder.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/analysis/peakfindsettings.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/analysis/waveformparser.h
    
    # Experiment configuration
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/chirpconfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/digitizerconfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/experiment.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/experimentobjective.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/experimentvalidator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/fid.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/ftmwconfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/ftmwconfigtypes.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/overlaybase.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/overlaytypes.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/catalogdata.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/rfconfig.h

    # Loadout system
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/loadout/rfconfigsnapshot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/loadout/chirpconfigloadout.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/loadout/ftmwdigitizerloadout.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/loadout/hardwareloadout.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/loadout/loadoutmanager.h

    # Hardware configuration classes (moved from hardware layer - pure data structures)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/hardware/core/ftmwdigitizerconfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/hardware/optional/pulsegenerator/pulsegenconfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/hardware/optional/flowcontroller/flowconfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/hardware/optional/ioboard/ioboardconfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/hardware/optional/pressurecontroller/pressurecontrollerconfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/experiment/hardware/optional/tempcontroller/temperaturecontrollerconfig.h
    
    # File parsers
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/processing/parsers/fileparser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/processing/parsers/catalogparser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/processing/parsers/fileparserregistry.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/processing/parsers/spcatparser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/processing/parsers/xiamparser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/processing/parsers/genericxyparser.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/processing/parsers/genericxydata.h
    
    # Data models moved to GUI layer - they depend on Qt widgets
    
    # Processing system
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/processing/overlayoperation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/processing/overlayprocessmanager.h
    
    # Presentation layer
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/presentation/curveappearance.h
    
    # Storage system
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/applicationconfigmanager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/auxdatastorage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/blackchirpcsv.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/datastoragebase.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/fidmultistorage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/fidpeakupstorage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/fidsinglestorage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/fidstoragebase.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/headerstorage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/overlaystorage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/settingsstorage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/storage/waveformbuffer.h
)

# ============================================================================
# LIF Data Components (Always included for runtime configuration)
# ============================================================================

# LIF data sources are always included for runtime configuration
list(APPEND BLACKCHIRP_DATA_SOURCES
    # LIF-specific data components
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/lif/lifconfig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/lif/lifstorage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/lif/liftrace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/lif/lifdigitizerconfig.cpp
)

list(APPEND BLACKCHIRP_DATA_HEADERS
    # LIF-specific data headers
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/lif/lifconfig.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/lif/lifstorage.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/lif/liftrace.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/lif/lifdigitizerconfig.h
)

# ============================================================================
# Platform-specific crash-handler source
# ============================================================================

if(WIN32)
    list(APPEND BLACKCHIRP_DATA_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/data/crashhandler_win.cpp
    )
else()
    list(APPEND BLACKCHIRP_DATA_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/data/crashhandler_unix.cpp
    )
endif()

# ============================================================================
# Create Data Library Target
# ============================================================================

# Create the blackchirp-data library
add_library(blackchirp-data STATIC
    ${BLACKCHIRP_DATA_SOURCES}
    ${BLACKCHIRP_DATA_HEADERS}
)

# Add alias for consistent naming
add_library(Blackchirp::Data ALIAS blackchirp-data)

# ============================================================================
# Target Properties and Configuration
# ============================================================================

# Set target properties
set_target_properties(blackchirp-data PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "blackchirp-data"
    EXPORT_NAME "Data"
)

# Include directories
target_include_directories(blackchirp-data
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
)

# ============================================================================
# Dependencies and Linking
# ============================================================================

# Qt6 dependencies. GSL is PUBLIC because ftworker.h (a public header
# of blackchirp-data) exposes <gsl/gsl_fft_real.h>, <gsl/gsl_interp.h>,
# and <gsl/gsl_spline.h>. Linking PRIVATE was invisible on Linux, where
# GSL lives in /usr/include and resolves through the compiler's default
# search path, but breaks on Windows with vcpkg, where the include dir
# only propagates via the IMPORTED target's INTERFACE_INCLUDE_DIRECTORIES.
target_link_libraries(blackchirp-data
    PUBLIC
        Qt6::Core
        Qt6::Gui
        Qt6::Network
        Qt6::Concurrent
        Eigen3::Eigen
        GSL::gsl
        GSL::gslcblas
)

# Add GSL math library on Unix systems
if(UNIX)
    target_link_libraries(blackchirp-data PRIVATE m)
endif()

# Crash-handler dependencies. Linux/glibc needs libdl for dladdr();
# libstdc++ 13+ requires -lstdc++exp for std::stacktrace symbol
# resolution. Windows needs dbghelp for MiniDumpWriteDump.
if(WIN32)
    target_link_libraries(blackchirp-data PRIVATE dbghelp)
else()
    target_link_libraries(blackchirp-data PRIVATE ${CMAKE_DL_LIBS})
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
            AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "13")
        target_link_libraries(blackchirp-data PRIVATE stdc++exp)
    endif()
endif()

# ============================================================================
# Compile Definitions
# ============================================================================

# Add version and configuration definitions
add_blackchirp_definitions(blackchirp-data)

# Data-layer specific definitions
target_compile_definitions(blackchirp-data PRIVATE
    BC_DATA_LIBRARY
)

# ============================================================================
# Installation Configuration
# ============================================================================

# Install library
install(TARGETS blackchirp-data
    EXPORT BlackchirpDataTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Libraries
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Libraries
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Applications
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/data/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/blackchirp/data
    COMPONENT Development
    FILES_MATCHING PATTERN "*.h"
)

# Export targets
install(EXPORT BlackchirpDataTargets
    FILE BlackchirpDataTargets.cmake
    NAMESPACE Blackchirp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Blackchirp
    COMPONENT Development
)

# ============================================================================
# Feature-Specific Configuration
# ============================================================================

# CUDA support (if enabled)
if(BC_ENABLE_CUDA)
    # Add CUDA-specific sources if they exist in the data layer
    # Currently no CUDA sources in data layer, but prepared for future
    target_link_libraries(blackchirp-data PRIVATE
        CUDA::cudart
    )
endif()

# Platform-specific configuration
if(WIN32)
    # Windows-specific data layer configuration
    target_compile_definitions(blackchirp-data PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
elseif(APPLE)
    # macOS-specific data layer configuration
    target_link_libraries(blackchirp-data PRIVATE
        ${COREFOUNDATION_LIBRARY}
    )
elseif(UNIX)
    # Linux-specific data layer configuration
    if(UDEV_FOUND)
        target_link_libraries(blackchirp-data PRIVATE ${UDEV_LIBRARIES})
        target_include_directories(blackchirp-data PRIVATE ${UDEV_INCLUDE_DIRS})
    endif()
endif()

# ============================================================================
# Development and Debugging
# ============================================================================

# Add compile options for better debugging
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(blackchirp-data PRIVATE
        BC_DEBUG
        QT_QML_DEBUG
    )
endif()

# Enable Qt logging for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(blackchirp-data PRIVATE
        QT_MESSAGELOGCONTEXT
    )
endif()

# ============================================================================
# Status Information
# ============================================================================

message(STATUS "Blackchirp Data Layer Configuration:")
message(STATUS "  Sources: ${BLACKCHIRP_DATA_SOURCES}")
message(STATUS "  Qt6 components: Core, Gui, Network")
message(STATUS "  GSL integration: ${GSL_FOUND}")
if(BC_ENABLE_CUDA)
    message(STATUS "  CUDA support: enabled")
endif()

# ============================================================================
# Internal Helper Functions
# ============================================================================

# Function to get data layer file lists (for use by other modules)
function(get_blackchirp_data_files SOURCES_VAR HEADERS_VAR)
    set(${SOURCES_VAR} ${BLACKCHIRP_DATA_SOURCES} PARENT_SCOPE)
    set(${HEADERS_VAR} ${BLACKCHIRP_DATA_HEADERS} PARENT_SCOPE)
endfunction()

# Function to check if data layer has specific features
function(blackchirp_data_has_feature FEATURE RESULT_VAR)
    set(${RESULT_VAR} FALSE PARENT_SCOPE)
    
    if(FEATURE STREQUAL "GSL")
        if(GSL_FOUND)
            set(${RESULT_VAR} TRUE PARENT_SCOPE)
        endif()
    elseif(FEATURE STREQUAL "CUDA")
        if(BC_ENABLE_CUDA)
            set(${RESULT_VAR} TRUE PARENT_SCOPE)
        endif()
    endif()
endfunction()