# BlackchirpViewerApplication.cmake - Blackchirp Viewer application target
#
# This module defines the blackchirp-viewer application executable target.
# It links together the data and viewer GUI libraries to create a minimal
# viewer application without hardware dependencies.

# Include guard to prevent multiple inclusions
if(BLACKCHIRP_VIEWER_APPLICATION_CMAKE_INCLUDED)
    return()
endif()
set(BLACKCHIRP_VIEWER_APPLICATION_CMAKE_INCLUDED TRUE)

# ============================================================================
# Viewer Application Source Files
# ============================================================================

set(BLACKCHIRP_VIEWER_APP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer-src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer-src/viewermainwindow.cpp
)

set(BLACKCHIRP_VIEWER_APP_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/viewer-src/viewermainwindow.h
)

# ============================================================================
# Resources
# ============================================================================

# Qt resources (shared with main application)
set(BLACKCHIRP_VIEWER_RESOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/resources/resources.qrc
)

# Process Qt resources
qt6_add_resources(BLACKCHIRP_VIEWER_RESOURCE_SOURCES ${BLACKCHIRP_VIEWER_RESOURCES})

# ============================================================================
# Viewer Application Target
# ============================================================================

# Create viewer executable
add_executable(blackchirp-viewer
    ${BLACKCHIRP_VIEWER_APP_SOURCES}
    ${BLACKCHIRP_VIEWER_APP_HEADERS}
    ${BLACKCHIRP_VIEWER_RESOURCE_SOURCES}
)

# Add alias for consistent naming
add_executable(Blackchirp::Viewer ALIAS blackchirp-viewer)

# Set target properties
set_target_properties(blackchirp-viewer PROPERTIES
    VERSION ${BCV_MAJOR_VERSION}.${BCV_MINOR_VERSION}.${BCV_PATCH_VERSION}
    OUTPUT_NAME "blackchirp-viewer"
    WIN32_EXECUTABLE TRUE
    MACOSX_BUNDLE TRUE
    EXPORT_NAME "Viewer"
)

# ============================================================================
# Dependencies and Linking
# ============================================================================

# Link required libraries (no SerialPort or Hardware for viewer)
target_link_libraries(blackchirp-viewer PRIVATE
    Blackchirp::Data
    Blackchirp::ViewerGui
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Network
    QWT::QWT
)

# Add system libraries if needed
if(UNIX)
    target_link_libraries(blackchirp-viewer PRIVATE m)
endif()

# ============================================================================
# Compile Definitions
# ============================================================================

# Add viewer-specific definitions
add_viewer_definitions(blackchirp-viewer)

# Also add standard Blackchirp definitions for data layer compatibility
add_blackchirp_definitions(blackchirp-viewer)

# ============================================================================
# Platform-Specific Configuration
# ============================================================================

if(WIN32)
    # Windows-specific configuration
    set_target_properties(blackchirp-viewer PROPERTIES
        WIN32_EXECUTABLE TRUE
    )
    
elseif(APPLE)
    # macOS bundle configuration
    set(_bcv_icns "${CMAKE_CURRENT_SOURCE_DIR}/icons/blackchirp.icns")
    set_target_properties(blackchirp-viewer PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/packaging/macos/ViewerInfo.plist"
        MACOSX_BUNDLE_BUNDLE_NAME "Blackchirp Viewer"
        MACOSX_BUNDLE_BUNDLE_VERSION ${BCV_MAJOR_VERSION}.${BCV_MINOR_VERSION}.${BCV_PATCH_VERSION}
        MACOSX_BUNDLE_SHORT_VERSION_STRING ${BCV_MAJOR_VERSION}.${BCV_MINOR_VERSION}.${BCV_PATCH_VERSION}
        MACOSX_BUNDLE_COPYRIGHT "Copyright © Kyle N. Crabtree"
        MACOSX_BUNDLE_ICON_FILE "blackchirp.icns"
    )
    if(EXISTS ${_bcv_icns})
        target_sources(blackchirp-viewer PRIVATE ${_bcv_icns})
        set_source_files_properties(${_bcv_icns} PROPERTIES
            MACOSX_PACKAGE_LOCATION "Resources"
        )
    endif()
    
else()
    # Linux desktop integration
    # TODO: Install viewer .desktop file and icons
endif()

# ============================================================================
# Installation
# ============================================================================

# Install viewer executable. BUNDLE DESTINATION `.` places the .app at the
# install-prefix root for DragNDrop DMG layout and for blackchirp_deploy_qt().
install(TARGETS blackchirp-viewer
    BUNDLE DESTINATION .
        COMPONENT Applications
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Applications
)

# Bundle Qt redistributables on Windows/macOS (no-op on Linux).
blackchirp_deploy_qt(blackchirp-viewer)

# ============================================================================
# Status Information
# ============================================================================

message(STATUS "Blackchirp Viewer Application:")
message(STATUS "  Executable: blackchirp-viewer")
message(STATUS "  Resources: ${BLACKCHIRP_VIEWER_RESOURCES}")
message(STATUS "  Data layer: linked")
message(STATUS "  Version: ${BCV_MAJOR_VERSION}.${BCV_MINOR_VERSION}.${BCV_PATCH_VERSION}")