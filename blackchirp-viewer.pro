#-------------------------------------------------
#
# BlackChirp Viewer - Data visualization application
# Based on BlackChirp by Kyle Crabtree
#
#-------------------------------------------------

QT       += core gui network widgets

TARGET = blackchirp-viewer
TEMPLATE = app

# Viewer-specific versioning
BCV_MAJOR_VERSION = 1
BCV_MINOR_VERSION = 0
BCV_PATCH_VERSION = 0
BCV_RELEASE_VERSION = alpha

DEFINES += BCV_MAJOR_VERSION=$$BCV_MAJOR_VERSION
DEFINES += BCV_MINOR_VERSION=$$BCV_MINOR_VERSION
DEFINES += BCV_PATCH_VERSION=$$BCV_PATCH_VERSION
DEFINES += BCV_RELEASE_VERSION=$$BCV_RELEASE_VERSION
DEFINES += BCV_BUILD_VERSION='$(shell cd $$PWD/src && git describe --always)'

# Define BC_VIEWER to enable conditional compilation
DEFINES += BC_VIEWER

CONFIG += c++latest strict_c++

SOURCES += $$PWD/viewer-src/main.cpp
RESOURCES += $$PWD/src/resources/resources.qrc
INCLUDEPATH += $$PWD/src

# Include viewer-specific dependencies
include($$PWD/viewer-src/data-viewer.pri)
include($$PWD/viewer-src/gui-viewer.pri)
include($$PWD/viewer-src/lif-viewer.pri)

OTHER_FILES += README.md

QMAKE_CXXFLAGS_RELEASE -= -O2
QMAKE_CXXFLAGS_RELEASE += -O3

# External dependencies
# QWT
win32 {
    QWT_PATH = C:/Qwt-6.2.0
    INCLUDEPATH += $${QWT_PATH}/src
    LIBS += -L$${QWT_PATH}/lib
}

!win32 {
    INCLUDEPATH += /usr/include/qwt
    LIBS += -lqwt-qt6
}

# GSL
win32 {
    GSL_PATH = C:/gsl-2.7
    INCLUDEPATH += $${GSL_PATH}/include
    LIBS += -L$${GSL_PATH}/lib -lgsl -lgslcblas
}

!win32 {
    INCLUDEPATH += /usr/include/gsl
    LIBS += -lgsl -lgslcblas
}