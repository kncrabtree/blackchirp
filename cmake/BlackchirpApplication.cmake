# BlackchirpApplication.cmake - Main Blackchirp application targets
#
# This module defines the main Blackchirp application executable target.
# It links together all the libraries (data, GUI, hardware) to create the
# complete application.

# Include guard to prevent multiple inclusions
if(BLACKCHIRP_APPLICATION_CMAKE_INCLUDED)
    return()
endif()
set(BLACKCHIRP_APPLICATION_CMAKE_INCLUDED TRUE)

# ============================================================================
# Main Application Source Files
# ============================================================================

set(BLACKCHIRP_APP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

set(BLACKCHIRP_APP_HEADERS
    # Main application doesn't have separate headers currently
    # All UI headers are in the GUI library
)

# ============================================================================
# Acquisition Layer Source Files
# ============================================================================

# The acquisition layer contains experiment management and batch processing
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
# Optional Module Source Files
# ============================================================================

# LIF Module (if enabled)
set(BLACKCHIRP_LIF_SOURCES)
set(BLACKCHIRP_LIF_HEADERS)

if(BC_LIF)
    set(BLACKCHIRP_LIF_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/modules/lif/data/lifconfig.cpp
        # Add other LIF sources as needed
    )
    
    set(BLACKCHIRP_LIF_HEADERS
        # Add LIF headers as needed
    )
endif()

# CUDA Module (if enabled)
set(BLACKCHIRP_CUDA_SOURCES)
set(BLACKCHIRP_CUDA_HEADERS)

if(BC_CUDA)
    set(BLACKCHIRP_CUDA_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/modules/cuda/gpuaverager.cu
        # Add other CUDA sources as needed
    )
    
    set(BLACKCHIRP_CUDA_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/modules/cuda/gpuaverager.h
        # Add other CUDA headers as needed
    )
endif()

# ============================================================================
# Resource Files
# ============================================================================

# ============================================================================
# Process Qt Resources
# ============================================================================

# Always include main resources
set(BLACKCHIRP_QRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/resources/resources.qrc
)

# Virtual FTMW digitizer no longer requires resource files

# Manually process Qt resources to ensure they are compiled
qt6_add_resources(BLACKCHIRP_COMPILED_RESOURCES ${BLACKCHIRP_QRC_FILES})

# ============================================================================
# Create Application Executable Target
# ============================================================================

# Combine all application sources
set(BLACKCHIRP_APP_ALL_SOURCES
    ${BLACKCHIRP_APP_SOURCES}
    ${BLACKCHIRP_ACQUISITION_SOURCES}
    ${BLACKCHIRP_LIF_SOURCES}
    ${BLACKCHIRP_CUDA_SOURCES}
)

set(BLACKCHIRP_APP_ALL_HEADERS
    ${BLACKCHIRP_APP_HEADERS}
    ${BLACKCHIRP_ACQUISITION_HEADERS}
    ${BLACKCHIRP_LIF_HEADERS}
    ${BLACKCHIRP_CUDA_HEADERS}
)

# Create the main Blackchirp executable
add_executable(blackchirp
    ${BLACKCHIRP_APP_ALL_SOURCES}
    ${BLACKCHIRP_APP_ALL_HEADERS}
    ${BLACKCHIRP_COMPILED_RESOURCES}
)

# Add alias for consistent naming
add_executable(Blackchirp::Application ALIAS blackchirp)

# ============================================================================
# Target Properties and Configuration
# ============================================================================

# Set target properties
set_target_properties(blackchirp PROPERTIES
    VERSION ${PROJECT_VERSION}
    OUTPUT_NAME "blackchirp"
    EXPORT_NAME "Application"
)

# Include directories
target_include_directories(blackchirp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}/src
)

# ============================================================================
# Dependencies and Linking
# ============================================================================

# Link all Blackchirp libraries
target_link_libraries(blackchirp
    PRIVATE
        Blackchirp::Data
        Blackchirp::Gui
        Blackchirp::Hardware
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        Qt6::Network
        GSL::gsl
        GSL::gslcblas
)

# Platform-specific libraries
if(UNIX)
    target_link_libraries(blackchirp PRIVATE m)
endif()

# Hardware-specific library linking (only link to final executable)
if(LABJACK_LIBRARY_PATH)
    message(STATUS "Linking LabJack library to main application: ${LABJACK_LIBRARY_PATH}")
    target_link_libraries(blackchirp PRIVATE ${LABJACK_LIBRARY_PATH})
endif()

# CUDA libraries (if enabled)
if(BC_CUDA AND CUDA_FOUND)
    target_link_libraries(blackchirp PRIVATE ${CUDA_LIBRARIES})
    # Set CUDA properties if needed
    set_target_properties(blackchirp PROPERTIES
        CUDA_STANDARD 11
        CUDA_STANDARD_REQUIRED ON
    )
endif()

# ============================================================================
# Compile Definitions
# ============================================================================

# Add version and configuration definitions
add_blackchirp_definitions(blackchirp)

# Application-specific definitions
target_compile_definitions(blackchirp PRIVATE
    BC_APPLICATION
    BC_MAIN_APPLICATION
)

# Module-specific definitions
if(BC_LIF)
    target_compile_definitions(blackchirp PRIVATE BC_LIF)
endif()

if(BC_CUDA)
    target_compile_definitions(blackchirp PRIVATE BC_CUDA)
endif()

# ============================================================================
# Installation Configuration
# ============================================================================

# Install the main application
install(TARGETS blackchirp
    EXPORT BlackchirpApplicationTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Applications
    BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Applications
)

# Export targets
install(EXPORT BlackchirpApplicationTargets
    FILE BlackchirpApplicationTargets.cmake
    NAMESPACE Blackchirp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Blackchirp
    COMPONENT Development
)

# ============================================================================
# Desktop Integration (Linux/Unix)
# ============================================================================

if(UNIX AND NOT APPLE)
    # Install desktop file for Linux desktop environments (if it exists)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/packaging/blackchirp.desktop.in)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/packaging/blackchirp.desktop.in
            ${CMAKE_CURRENT_BINARY_DIR}/blackchirp.desktop
            @ONLY
        )
        
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/blackchirp.desktop
            DESTINATION share/applications
            COMPONENT Applications
        )
    endif()
    
    # Install application icon
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/resources/icons/bc_logo_large.png)
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/resources/icons/bc_logo_large.png
            DESTINATION share/pixmaps
            RENAME blackchirp.png
            COMPONENT Applications
        )
    endif()
endif()

# ============================================================================
# Status Information
# ============================================================================

message(STATUS "Blackchirp Application Configuration:")
message(STATUS "  Main executable: blackchirp")
message(STATUS "  Dependencies: Data, GUI, Hardware libraries")
message(STATUS "  LIF module: ${BC_LIF}")
message(STATUS "  CUDA module: ${BC_CUDA}")
message(STATUS "  Version: ${PROJECT_VERSION}")