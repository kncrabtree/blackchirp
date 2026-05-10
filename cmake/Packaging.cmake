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

    # NSIS specific settings.
    #
    # Forward slashes are used in path-shaped values rather than escaped
    # backslashes. CPack writes these into the auto-generated
    # CPackConfig.cmake as quoted strings; with backslashes, CMake parses
    # the resulting `\P` / `\a` / `\b` as invalid escape sequences and
    # emits CMP0010 dev warnings. NSIS itself accepts forward slashes in
    # icon and InstallDir contexts, and the install-root value uses
    # NSIS's `$PROGRAMFILES64` predefined variable rather than a literal
    # path so it follows the user's localized Program Files location.
    set(CPACK_NSIS_DISPLAY_NAME "Blackchirp")
    set(CPACK_NSIS_PACKAGE_NAME "Blackchirp")
    set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")

    set(CPACK_NSIS_MUI_ICON "${CMAKE_CURRENT_SOURCE_DIR}/icons/blackchirp.ico")
    set(CPACK_NSIS_MUI_UNIICON "${CMAKE_CURRENT_SOURCE_DIR}/icons/blackchirp.ico")
    # Path under the install prefix to the executable whose icon is shown
    # in Windows' "Add or Remove Programs" / "Apps & features" entry.
    set(CPACK_NSIS_INSTALLED_ICON_NAME "bin/blackchirp.exe")
    
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

    # ------------------------------------------------------------------
    # Bundle non-Qt third-party runtime DLLs alongside the executables.
    # windeployqt walks Qt module imports only; qwt.dll, gsl.dll,
    # gslcblas.dll, and any other third-party DLL the .exe depends on
    # are not copied automatically. Without these the installed binary
    # fails to launch with STATUS_DLL_NOT_FOUND, because the PE loader
    # search order (.exe directory → CWD → System32 → PATH) does not
    # include the upstream build trees.
    # ------------------------------------------------------------------
    set(_bc_runtime_dlls)

    # Qwt (from-source build via qmake; `nmake install` lays out qwt.dll
    # under qwt-install/lib/ on Windows rather than the more common bin/
    # convention).
    if(QWT_LIBRARY)
        get_filename_component(_qwt_lib_dir "${QWT_LIBRARY}" DIRECTORY)
        foreach(_candidate "${_qwt_lib_dir}/qwt.dll"
                           "${_qwt_lib_dir}/../bin/qwt.dll")
            if(EXISTS "${_candidate}")
                get_filename_component(_resolved "${_candidate}" ABSOLUTE)
                list(APPEND _bc_runtime_dlls "${_resolved}")
                break()
            endif()
        endforeach()
        unset(_qwt_lib_dir)
    endif()

    # GSL (vcpkg's x64-windows triplet, dynamic). CMake's FindGSL sets
    # IMPORTED_LOCATION to the .lib it finds via find_library; the
    # paired .dll lives in the sibling bin/ under vcpkg's installed
    # tree. Resolve it by walking back from the import library.
    foreach(_gsl_target GSL::gsl GSL::gslcblas)
        if(TARGET ${_gsl_target})
            get_target_property(_lib ${_gsl_target} IMPORTED_LOCATION_RELEASE)
            if(NOT _lib)
                get_target_property(_lib ${_gsl_target} IMPORTED_LOCATION)
            endif()
            if(_lib)
                get_filename_component(_lib_dir "${_lib}" DIRECTORY)
                get_filename_component(_lib_we  "${_lib}" NAME_WE)
                foreach(_candidate "${_lib_dir}/${_lib_we}.dll"
                                   "${_lib_dir}/../bin/${_lib_we}.dll")
                    if(EXISTS "${_candidate}")
                        get_filename_component(_resolved "${_candidate}" ABSOLUTE)
                        list(APPEND _bc_runtime_dlls "${_resolved}")
                        break()
                    endif()
                endforeach()
            endif()
            unset(_lib)
            unset(_lib_dir)
            unset(_lib_we)
        endif()
    endforeach()

    if(_bc_runtime_dlls)
        list(REMOVE_DUPLICATES _bc_runtime_dlls)
        install(FILES ${_bc_runtime_dlls}
            DESTINATION "${CMAKE_INSTALL_BINDIR}"
            COMPONENT Applications)
        message(STATUS "Bundling Windows runtime DLLs:")
        foreach(_dll ${_bc_runtime_dlls})
            message(STATUS "  ${_dll}")
        endforeach()
    else()
        message(WARNING
            "No third-party runtime DLLs located for Windows packaging; "
            "the installed .exe is likely to fail with STATUS_DLL_NOT_FOUND. "
            "Check QWT_LIBRARY and GSL::gsl IMPORTED_LOCATION.")
    endif()
    unset(_bc_runtime_dlls)

elseif(APPLE)
    # ========================================================================
    # macOS Packaging (DragNDrop DMG)
    # ========================================================================

    list(APPEND CPACK_GENERATOR "DragNDrop" "TGZ")

    # DMG settings
    set(CPACK_DMG_VOLUME_NAME "Blackchirp")
    set(CPACK_DMG_FORMAT "UDZO")

    # Skip the DMG software-license agreement. The DragNDrop generator
    # turns CPACK_RESOURCE_FILE_LICENSE (set globally above) into an SLA
    # resource that hdiutil prompts the user to accept on every mount.
    # That blocks non-interactive CI smoke tests — hdiutil prints the
    # license, fails to read a Y/N answer from a runner with no TTY, and
    # returns `attach canceled` — and is unusual for permissively-
    # licensed macOS apps where the license lives inside the .app bundle
    # already. The deb/rpm/NSIS packages keep their license dialog via
    # the generic CPACK_RESOURCE_FILE_LICENSE; CPACK_DMG_SLA_USE_RESOURCE_FILE_LICENSE
    # is the per-generator opt-in that the DragNDrop generator gates on.
    set(CPACK_DMG_SLA_USE_RESOURCE_FILE_LICENSE OFF)

    # Bundle metadata is set on the executable targets via MACOSX_BUNDLE_*
    # properties (see BlackchirpApplication.cmake / BlackchirpViewerApplication.cmake).
    # The DragNDrop generator picks those up automatically; no CPACK_BUNDLE_*
    # variables are needed (those apply to the separate Bundle generator).

    # Do NOT set CPACK_SET_DESTDIR here. DESTDIR-style staging is the right
    # mode for system package generators (DEB/RPM/IFW) where the package
    # must record `/usr` as the install root. For DragNDrop the .app *is*
    # the unit of distribution and the package root is the install root;
    # with DESTDIR=ON, CPack would stage the .app under
    # `${DESTDIR}/usr/local/blackchirp.app`, the install(CODE) hook in
    # QtDeployment.cmake would look for `${CMAKE_INSTALL_PREFIX}/blackchirp.app`
    # (i.e., `/usr/local/blackchirp.app`) and miss it, and the DragNDrop
    # generator's own file walk would not pick the .app into the dmg.

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
# per-target cmake modules). Only Applications ships in release packages;
# all blackchirp-* libraries are STATIC and linked into the executables, so
# the Libraries/Development install rules exist only for `cmake --install`
# in source-tree workflows and add nothing useful to a binary package.
set(CPACK_COMPONENTS_ALL Applications)

# CPACK_COMPONENTS_ALL is silently ignored for the DEB / RPM / ARCHIVE /
# NSIS generators unless component-based packaging is explicitly enabled,
# so without these the binary packages bundle every install rule (headers,
# static libs, CMake export files) regardless of the Applications-only
# intent above. Release builds compile with `-O3 -g` for crash-symbol
# capture, and CPACK_STRIP_FILES strips executables but not the `.a`/`.lib`
# archives, so an uncomponentized package balloons (the Windows .zip was
# 88 MB compressed with 266 MB of static archives in lib/ before this
# filter). Per-component packaging restricts the binary to the
# Applications component on every generator we target.
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_NSIS_COMPONENT_INSTALL ON)

# Drop the per-component "-Applications" suffix that component-install adds
# to file names; we only ship one component, so the canonical
# `Blackchirp-<version>-<system>-<arch>.{deb,rpm,zip,tar.gz}` form is what
# the workflow's `Blackchirp-*.{deb,rpm,zip,tar.gz}` upload-globs expect.
set(CPACK_DEBIAN_APPLICATIONS_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}.deb")
set(CPACK_RPM_APPLICATIONS_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}.rpm")
set(CPACK_ARCHIVE_APPLICATIONS_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}")

# The on-disk filename and the installed-package identity are separate.
# CPack's per-component default identity is `${CPACK_PACKAGE_NAME}-${COMP}`
# ("blackchirp-applications" / "Blackchirp-Applications"), which is what
# `dpkg -l` / `rpm -q` would show. Override to the bare project name —
# the only component we ship is Applications, so the suffix conveys
# nothing and confuses package-manager tooling.
set(CPACK_DEBIAN_APPLICATIONS_PACKAGE_NAME "blackchirp")
set(CPACK_RPM_APPLICATIONS_PACKAGE_NAME "blackchirp")

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

# ============================================================================
# Optional: bundle Qwt's shared library inside the package
# ============================================================================

# No Linux distribution currently ships a Qt6-built Qwt 6.x in its main
# repos that Ubuntu/Debian deb tooling can resolve (Ubuntu has only the
# Qt5-era qwt 6.1.4). When BC_BUNDLE_QWT is ON, the package ships
# libqwt.so* in a private subdir of the libdir and the executables get
# an RPATH so they find it at runtime. Off by default; enabled by the
# CI deb job and by anyone packaging on a distro without a Qt6 Qwt
# package. openSUSE has libqwt6-qt6-6_3 in its standard repos, so the
# rpm job leaves this off and links against the system Qwt instead.
option(BC_BUNDLE_QWT
    "Install Qwt's shared library inside the package and add an RPATH \
to the executables so they find it. Use on Linux distros that lack a \
Qt6-compatible Qwt package."
    OFF)

if(BC_BUNDLE_QWT)
    if(NOT QWT_LIBRARY)
        message(FATAL_ERROR
            "BC_BUNDLE_QWT=ON requires Qwt to have been found "
            "(QWT_LIBRARY is unset).")
    endif()
    if(NOT (UNIX AND NOT APPLE))
        message(FATAL_ERROR
            "BC_BUNDLE_QWT is only meaningful on Linux package generators.")
    endif()

    # Install the realfile and every SONAME symlink. dpkg-shlibdeps and
    # ld.so look at the binary's NEEDED entry (libqwt.so.<v>), not at the
    # bare libqwt.so, so the symlinks must travel with the realfile.
    # Derive the lib basename from QWT_LIBRARY rather than hard-coding
    # `libqwt`: the from-source qmake build produces libqwt.so* but
    # distros ship distinct Qt5/Qt6 libs (e.g., openSUSE's libqwt-qt6.so).
    get_filename_component(_bc_qwt_lib_dir "${QWT_LIBRARY}" DIRECTORY)
    get_filename_component(_bc_qwt_lib_name "${QWT_LIBRARY}" NAME)
    string(REGEX REPLACE "\\.so.*$" "" _bc_qwt_lib_base "${_bc_qwt_lib_name}")
    file(GLOB _bc_qwt_lib_files
        "${_bc_qwt_lib_dir}/${_bc_qwt_lib_base}.so"
        "${_bc_qwt_lib_dir}/${_bc_qwt_lib_base}.so.*")
    if(NOT _bc_qwt_lib_files)
        message(FATAL_ERROR
            "BC_BUNDLE_QWT=ON but no ${_bc_qwt_lib_base}.so* found "
            "under ${_bc_qwt_lib_dir}.")
    endif()
    install(FILES ${_bc_qwt_lib_files}
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/blackchirp"
        COMPONENT Applications)

    # RPATH so the executables resolve the bundled libqwt at runtime.
    # $ORIGIN is interpreted by ld.so as the directory containing the
    # binary, so the relative jump out of bin/ and into the libdir is
    # what reaches the bundle from /usr/bin/blackchirp.
    foreach(_bc_app blackchirp blackchirp-viewer)
        if(TARGET ${_bc_app})
            set_target_properties(${_bc_app} PROPERTIES
                INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/blackchirp"
                BUILD_WITH_INSTALL_RPATH OFF)
        endif()
    endforeach()
endif()

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