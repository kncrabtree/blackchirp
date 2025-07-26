# Main Blackchirp Application CMakeLists.txt
#
# This builds the main blackchirp application with full hardware support

# Only build main application if not viewer-only build
if(BC_BUILD_VIEWER_ONLY)
    return()
endif()

# ============================================================================
# Include Component Modules
# ============================================================================

# Data layer already created in root CMakeLists.txt

# TODO: Include other layers as they are converted
# include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/BlackchirpGui.cmake)
# include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/BlackchirpHardware.cmake)
# include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/BlackchirpAcquisition.cmake)

# ============================================================================
# Main Application Sources
# ============================================================================

set(BLACKCHIRP_MAIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

# ============================================================================
# Resources
# ============================================================================

# Qt resources
set(BLACKCHIRP_RESOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/resources.qrc
)

# Process Qt resources
qt6_add_resources(BLACKCHIRP_RESOURCE_SOURCES ${BLACKCHIRP_RESOURCES})

# ============================================================================
# Main Application Target
# ============================================================================

# Create main executable
add_executable(blackchirp
    ${BLACKCHIRP_MAIN_SOURCES}
    ${BLACKCHIRP_RESOURCE_SOURCES}
)

# Set target properties
set_target_properties(blackchirp PROPERTIES
    OUTPUT_NAME "blackchirp"
    WIN32_EXECUTABLE TRUE
    MACOSX_BUNDLE TRUE
)

# ============================================================================
# Dependencies and Linking
# ============================================================================

# Link required libraries
target_link_libraries(blackchirp PRIVATE
    Blackchirp::Data
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Network
    Qt6::SerialPort
)

# Add version and configuration definitions
add_blackchirp_definitions(blackchirp)

# ============================================================================
# Installation
# ============================================================================

# Install main executable
install(TARGETS blackchirp
    BUNDLE DESTINATION .
        COMPONENT Applications
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Applications
)

# ============================================================================
# Platform-Specific Configuration
# ============================================================================

if(WIN32)
    # Windows-specific configuration
    set_target_properties(blackchirp PROPERTIES
        WIN32_EXECUTABLE TRUE
    )
    
    # TODO: Add Windows-specific resources (icon, version info)
    
elseif(APPLE)
    # macOS bundle configuration
    set_target_properties(blackchirp PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/../packaging/macos/Info.plist"
        MACOSX_BUNDLE_BUNDLE_NAME "Blackchirp"
        MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
        MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION}
        MACOSX_BUNDLE_COPYRIGHT "Copyright © 2023 CrabtreeLab"
    )
    
else()
    # Linux desktop integration
    # TODO: Install .desktop file and icons
endif()

# ============================================================================
# Status Information
# ============================================================================

message(STATUS "Main Blackchirp Application:")
message(STATUS "  Executable: blackchirp")
message(STATUS "  Resources: ${BLACKCHIRP_RESOURCES}")
message(STATUS "  Data layer: linked")
# TODO: Add status for other layers as they are added