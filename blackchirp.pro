#-------------------------------------------------
#
# Project created by QtCreator 2015-02-11T14:07:58
#
#-------------------------------------------------

QT       += core gui network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets serialport

TARGET = blackchirp
TEMPLATE = app

BC_MAJOR_VERSION = 1
BC_MINOR_VERSION = 0
BC_PATCH_VERSION = 0
BC_RELEASE_VERSION = release

DEFINES += BC_MAJOR_VERSION=$$BC_MAJOR_VERSION
DEFINES += BC_MINOR_VERSION=$$BC_MINOR_VERSION
DEFINES += BC_PATCH_VERSION=$$BC_PATCH_VERSION
DEFINES += BC_RELEASE_VERSION=$$BC_RELEASE_VERSION
DEFINES += BC_BUILD_VERSION='$(shell cd $$PWD/src && git describe --always)'

CONFIG += c++latest strict_c++

SOURCES += $$PWD/src/main.cpp
RESOURCES += $$PWD/src/resources/resources.qrc
INCLUDEPATH += $$PWD/src

!exists($$PWD/src/config/config.pri) {
     error('config.pri file not found in src/config. Please copy src/config/config.pri.template to src/config/config.pri, and then edit src/config/config.pri as needed.')
}
include($$PWD/src/config/config.pri)
include($$PWD/src/acquisition/acquisition.pri)
include($$PWD/src/data/data.pri)
include($$PWD/doc/doc.pri)
include($$PWD/src/gui/gui.pri)
include($$PWD/src/hardware/hardware.pri)

gpu-cuda {
  include($$PWD/src/modules/cuda/cuda.pri)
}

lif {
  include($$PWD/src/modules/lif/lif.pri)
}

OTHER_FILES += README.md \
               $$PWD/src/config/config.pri.template

QMAKE_CXXFLAGS_RELEASE -= -O2
QMAKE_CXXFLAGS_RELEASE += -O3

DISTFILES += \
    CONTRIBUTING.md \
    changelog.md


