# FindQWT.cmake - Find QWT (Qt Widgets for Technical Applications)
#
# This module defines:
#  QWT_FOUND - system has QWT
#  QWT_INCLUDE_DIRS - the QWT include directories
#  QWT_LIBRARIES - the libraries needed to use QWT
#  QWT_VERSION - the version of QWT found
#  QWT::QWT - imported target for QWT

find_package(PkgConfig QUIET)

# Try pkg-config first
if(PkgConfig_FOUND)
    pkg_check_modules(PC_QWT QUIET qwt)
endif()

# Find include directory
find_path(QWT_INCLUDE_DIR
    NAMES qwt.h
    HINTS 
        ${PC_QWT_INCLUDE_DIRS}
        ${QWT_ROOT}/include
        $ENV{QWT_ROOT}/include
    PATHS
        /usr/include/qwt
        /usr/include/qwt-qt6
        /usr/include/qt6/qwt
        /usr/include/qt6/qwt6
        /usr/include/qwt6
        /usr/local/include/qwt
        /opt/local/include/qwt
        "C:/Qwt-6.3.0/include"
        "C:/Qwt-6.2.0/include"
        "C:/Program Files/Qwt/include"
    PATH_SUFFIXES
        qwt
        qwt-qt6
)

# Find library
find_library(QWT_LIBRARY
    NAMES qwt qwt-qt6 qwt6
    HINTS 
        ${PC_QWT_LIBRARY_DIRS}
        ${QWT_ROOT}/lib
        $ENV{QWT_ROOT}/lib
    PATHS
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /opt/local/lib
        "C:/Qwt-6.3.0/lib"
        "C:/Qwt-6.2.0/lib"
        "C:/Program Files/Qwt/lib"
)

# Extract version from qwt_global.h
if(QWT_INCLUDE_DIR AND EXISTS "${QWT_INCLUDE_DIR}/qwt_global.h")
    file(READ "${QWT_INCLUDE_DIR}/qwt_global.h" QWT_GLOBAL_H)
    
    string(REGEX MATCH "#define[ \t]+QWT_VERSION[ \t]+0x([0-9A-Fa-f]+)" 
           QWT_VERSION_MATCH "${QWT_GLOBAL_H}")
    
    if(QWT_VERSION_MATCH)
        set(QWT_VERSION_HEX ${CMAKE_MATCH_1})
        
        # Convert hex version to decimal components
        math(EXPR QWT_VERSION_MAJOR "(0x${QWT_VERSION_HEX} >> 16) & 0xFF")
        math(EXPR QWT_VERSION_MINOR "(0x${QWT_VERSION_HEX} >> 8) & 0xFF")
        math(EXPR QWT_VERSION_PATCH "0x${QWT_VERSION_HEX} & 0xFF")
        
        set(QWT_VERSION "${QWT_VERSION_MAJOR}.${QWT_VERSION_MINOR}.${QWT_VERSION_PATCH}")
    endif()
endif()

# Use FindPackageHandleStandardArgs
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(QWT
    REQUIRED_VARS QWT_LIBRARY QWT_INCLUDE_DIR
    VERSION_VAR QWT_VERSION
)

if(QWT_FOUND)
    set(QWT_LIBRARIES ${QWT_LIBRARY})
    set(QWT_INCLUDE_DIRS ${QWT_INCLUDE_DIR})
    
    # Create imported target
    if(NOT TARGET QWT::QWT)
        add_library(QWT::QWT UNKNOWN IMPORTED)
        set_target_properties(QWT::QWT PROPERTIES
            IMPORTED_LOCATION "${QWT_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${QWT_INCLUDE_DIR}"
        )
        
        # QWT requires Qt widgets
        find_package(Qt6 REQUIRED COMPONENTS Widgets)
        set_target_properties(QWT::QWT PROPERTIES
            INTERFACE_LINK_LIBRARIES "Qt6::Widgets"
        )
    endif()
endif()

mark_as_advanced(QWT_INCLUDE_DIR QWT_LIBRARY)