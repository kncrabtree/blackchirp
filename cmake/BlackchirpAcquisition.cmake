# BlackchirpAcquisition.cmake - Acquisition layer for Blackchirp
#
# This module defines the blackchirp-acquisition library target containing:
# - AcquisitionManager, which drives a running experiment
# - The batch managers that sequence experiments
#
# The layer sits between the GUI and the data layer: MainWindow owns an
# AcquisitionManager and the batch managers, and nothing under
# src/acquisition/ includes anything from src/gui/. Keeping it in its own
# library (rather than compiling it straight into the blackchirp
# executable) is what makes blackchirp-gui self-contained enough to link
# into a test target.

# Include guard to prevent multiple inclusions
if(BLACKCHIRP_ACQUISITION_CMAKE_INCLUDED)
    return()
endif()
set(BLACKCHIRP_ACQUISITION_CMAKE_INCLUDED TRUE)

# ============================================================================
# Acquisition Layer Source Files
# ============================================================================

set(BLACKCHIRP_ACQUISITION_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/acquisition/acquisitionmanager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/acquisition/batch/batchmanager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/acquisition/batch/batchsequence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/acquisition/batch/batchsingle.cpp
)

set(BLACKCHIRP_ACQUISITION_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/acquisition/acquisitionmanager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/acquisition/batch/batchmanager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/acquisition/batch/batchsequence.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/acquisition/batch/batchsingle.h
)

# ============================================================================
# Create Acquisition Library Target
# ============================================================================

add_library(blackchirp-acquisition STATIC
    ${BLACKCHIRP_ACQUISITION_SOURCES}
    ${BLACKCHIRP_ACQUISITION_HEADERS}
)

# Add alias for consistent naming
add_library(Blackchirp::Acquisition ALIAS blackchirp-acquisition)

# ============================================================================
# Target Properties and Configuration
# ============================================================================

set_target_properties(blackchirp-acquisition PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "blackchirp-acquisition"
    EXPORT_NAME "Acquisition"
)

target_include_directories(blackchirp-acquisition
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
)

# ============================================================================
# Dependencies and Linking
# ============================================================================

target_link_libraries(blackchirp-acquisition
    PUBLIC
        Qt6::Core
        Qt6::Concurrent
        Blackchirp::Data
)

# ============================================================================
# Compile Definitions
# ============================================================================

add_blackchirp_definitions(blackchirp-acquisition)

# ============================================================================
# Installation Configuration
# ============================================================================

install(TARGETS blackchirp-acquisition
    EXPORT BlackchirpAcquisitionTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Libraries
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Libraries
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Applications
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/acquisition/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/blackchirp/acquisition
    COMPONENT Development
    FILES_MATCHING PATTERN "*.h"
)

install(EXPORT BlackchirpAcquisitionTargets
    FILE BlackchirpAcquisitionTargets.cmake
    NAMESPACE Blackchirp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Blackchirp
    COMPONENT Development
)

# ============================================================================
# Status Information
# ============================================================================

message(STATUS "Blackchirp Acquisition Layer Configuration:")
message(STATUS "  Qt6 components: Core, Concurrent")
message(STATUS "  Dependencies: Data library")
