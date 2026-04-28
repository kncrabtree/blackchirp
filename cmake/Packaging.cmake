# Packaging.cmake - CPack configuration for Blackchirp
#
# This module configures CPack for cross-platform packaging

# Only configure packaging if this is the main project
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    return()
endif()

# ============================================================================
# General Package Settings
# ============================================================================

set(CPACK_PACKAGE_NAME "Blackchirp")
set(CPACK_PACKAGE_VENDOR "Kyle N. Crabtree")
set(CPACK_PACKAGE_CONTACT "Kyle N. Crabtree <kncrabtree@ucdavis.edu>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CP-FTMW Spectroscopy Software")
set(CPACK_PACKAGE_DESCRIPTION 
    "Blackchirp is open-source data acquisition software for CP-FTMW "
    "(Chirped-Pulse Fourier Transform Microwave) spectroscopy. It provides "
    "comprehensive control of spectroscopy hardware and data analysis tools.")

# Version information
set(CPACK_PACKAGE_VERSION_MAJOR ${BC_MAJOR_VERSION})
set(CPACK_PACKAGE_VERSION_MINOR ${BC_MINOR_VERSION})
set(CPACK_PACKAGE_VERSION_PATCH ${BC_PATCH_VERSION})
set(CPACK_PACKAGE_VERSION "${BC_MAJOR_VERSION}.${BC_MINOR_VERSION}.${BC_PATCH_VERSION}")

# Package file name
set(CPACK_PACKAGE_FILE_NAME 
    "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

# License and readme
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/COPYING")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# ============================================================================
# Platform-Specific Configuration
# ============================================================================

if(WIN32)
    # ========================================================================
    # Windows Packaging (NSIS)
    # ========================================================================
    
    list(APPEND CPACK_GENERATOR "NSIS" "ZIP")
    
    # NSIS specific settings
    set(CPACK_NSIS_DISPLAY_NAME "Blackchirp")
    set(CPACK_NSIS_PACKAGE_NAME "Blackchirp")
    set(CPACK_NSIS_INSTALL_ROOT "C:\\Program Files")
    
    # Start menu and desktop shortcuts
    set(CPACK_NSIS_MENU_LINKS
        "bin/blackchirp.exe" "Blackchirp"
        "bin/blackchirp-viewer.exe" "Blackchirp Viewer"
    )
    
    set(CPACK_NSIS_CREATE_ICONS_EXTRA
        "CreateShortCut '$DESKTOP\\\\Blackchirp.lnk' '$INSTDIR\\\\bin\\\\blackchirp.exe'"
        "CreateShortCut '$DESKTOP\\\\Blackchirp Viewer.lnk' '$INSTDIR\\\\bin\\\\blackchirp-viewer.exe'"
    )
    
    set(CPACK_NSIS_DELETE_ICONS_EXTRA
        "Delete '$DESKTOP\\\\Blackchirp.lnk'"
        "Delete '$DESKTOP\\\\Blackchirp Viewer.lnk'"
    )
    
    # Uninstaller
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
    
    # Registry entries for file associations (optional)
    # set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS
    #     "WriteRegStr HKCR '.bc' '' 'BlackchirpFile'"
    # )
    
elseif(APPLE)
    # ========================================================================
    # macOS Packaging (DragNDrop DMG)
    # ========================================================================

    list(APPEND CPACK_GENERATOR "DragNDrop" "TGZ")

    # DMG settings
    set(CPACK_DMG_VOLUME_NAME "Blackchirp")
    set(CPACK_DMG_FORMAT "UDZO")

    # Bundle metadata is set on the executable targets via MACOSX_BUNDLE_*
    # properties (see BlackchirpApplication.cmake / BlackchirpViewerApplication.cmake).
    # The DragNDrop generator picks those up automatically; no CPACK_BUNDLE_*
    # variables are needed (those apply to the separate Bundle generator).

    # macOS specific installation
    set(CPACK_SET_DESTDIR TRUE)

else()
    # ========================================================================
    # Linux Packaging (DEB, RPM, TGZ)
    # ========================================================================
    
    list(APPEND CPACK_GENERATOR "DEB" "RPM" "TGZ")
    
    # DEB package settings
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Kyle Crabtree <kncrabtree@ucdavis.edu>")
    set(CPACK_DEBIAN_PACKAGE_SECTION "science")
    set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
    set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/kncrabtree/blackchirp")

    # Debian dependencies are derived from linked .so files via SHLIBDEPS
    # (see CPACK_DEBIAN_PACKAGE_SHLIBDEPS below). Avoid hard-coding distro-
    # specific package names, which drift across Ubuntu/Debian releases.

    # Desktop integration
    set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
        "${CMAKE_CURRENT_SOURCE_DIR}/packaging/linux/postinst"
        "${CMAKE_CURRENT_SOURCE_DIR}/packaging/linux/prerm"
    )

    # RPM package settings
    set(CPACK_RPM_PACKAGE_SUMMARY "CP-FTMW Spectroscopy Software")
    set(CPACK_RPM_PACKAGE_GROUP "Applications/Science")
    set(CPACK_RPM_PACKAGE_LICENSE "GPLv3+")
    set(CPACK_RPM_PACKAGE_URL "https://github.com/kncrabtree/blackchirp")
    set(CPACK_RPM_PACKAGE_RELOCATABLE ON)

    # RPM dependencies are derived from linked .so files via AUTOREQ (set
    # below). This keeps the package working across openSUSE, Fedora, and
    # RHEL without naming drift between their qt6/gsl/qwt packages.
    
    # Architecture detection
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
        set(CPACK_RPM_PACKAGE_ARCHITECTURE "x86_64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i[3-6]86")
        set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "i386")
        set(CPACK_RPM_PACKAGE_ARCHITECTURE "i686")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
        set(CPACK_RPM_PACKAGE_ARCHITECTURE "aarch64")
    endif()
    
endif()

# ============================================================================
# Component-based packaging
# ============================================================================

# Define components (names match install(... COMPONENT ...) calls in the
# per-target cmake modules)
set(CPACK_COMPONENTS_ALL Applications Libraries Development)

# Application component
set(CPACK_COMPONENT_APPLICATIONS_DISPLAY_NAME "Applications")
set(CPACK_COMPONENT_APPLICATIONS_DESCRIPTION
    "Blackchirp main application and viewer")
set(CPACK_COMPONENT_APPLICATIONS_REQUIRED TRUE)

# Libraries component (for development)
set(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "Runtime Libraries")
set(CPACK_COMPONENT_LIBRARIES_DESCRIPTION
    "Shared libraries required for Blackchirp")
set(CPACK_COMPONENT_LIBRARIES_DEPENDS Applications)

# Development component
set(CPACK_COMPONENT_DEVELOPMENT_DISPLAY_NAME "Development Files")
set(CPACK_COMPONENT_DEVELOPMENT_DESCRIPTION
    "Header files and development libraries")
set(CPACK_COMPONENT_DEVELOPMENT_OPTIONAL TRUE)
set(CPACK_COMPONENT_DEVELOPMENT_DEPENDS Libraries)

# ============================================================================
# Source packaging
# ============================================================================

set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_IGNORE_FILES
    "/\\\\.git/"
    "/\\\\.gitignore"
    "/build.*/"
    "/\\\\.vscode/"
    "/\\\\.idea/"
    ".*\\\\.user$"
    ".*\\\\.swp$"
    ".*\\\\.orig$"
    "/CMakeFiles/"
    "/CMakeCache\\\\.txt$"
    "/cmake_install\\\\.cmake$"
    "/Makefile$"
    "/moc_.*\\\\.cpp$"
    "/ui_.*\\\\.h$"
    "/qrc_.*\\\\.cpp$"
)

# ============================================================================
# Advanced packaging options
# ============================================================================

# Strip binaries in release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CPACK_STRIP_FILES TRUE)
endif()

# Multi-config generators
set(CPACK_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo")

# Package dependencies
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_RPM_PACKAGE_AUTOREQ ON)

# Include CPack
include(CPack)

# ============================================================================
# Custom packaging targets
# ============================================================================

# Add custom target for creating all packages
add_custom_target(package-all
    COMMAND ${CMAKE_CPACK_COMMAND} --config CPackConfig.cmake
    COMMAND ${CMAKE_CPACK_COMMAND} --config CPackSourceConfig.cmake
    COMMENT "Creating all packages (binary and source)"
)

# Platform-specific package targets
if(WIN32)
    add_custom_target(package-nsis
        COMMAND ${CMAKE_CPACK_COMMAND} -G NSIS
        COMMENT "Creating NSIS installer"
    )
elseif(APPLE)
    add_custom_target(package-dmg
        COMMAND ${CMAKE_CPACK_COMMAND} -G DragNDrop
        COMMENT "Creating DMG package"
    )
else()
    add_custom_target(package-deb
        COMMAND ${CMAKE_CPACK_COMMAND} -G DEB
        COMMENT "Creating DEB package"
    )
    
    add_custom_target(package-rpm
        COMMAND ${CMAKE_CPACK_COMMAND} -G RPM
        COMMENT "Creating RPM package"
    )
endif()