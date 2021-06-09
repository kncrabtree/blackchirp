#-------------------------------------------------
#
# Project created by QtCreator 2015-02-11T14:07:58
#
#-------------------------------------------------

#This project is only set up to work properly on UNIX, because the default
#save path starts with /. It could be easily modified to run on windows; simply
#modify savePath and appDataPath in main.cpp. Both directories need to be
#writable by the user running the program.
#
#This version allows compile-time selection of hardware, and also has virtual
#hardware implementations so that it can run on systems not connected to real
#instrumentation. In order to compile this program, you must copy the
#config.pri.template file to config.pri. config.pri will be ignored by git, and
#can safely be used to store the local configuration. Do not modify
#config.pri.template unless you are making changes relevant for all instruments
#(eg adding new hardware).

QT       += core gui network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets serialport

TARGET = blackchirp
TEMPLATE = app

CONFIG += c++17

SOURCES += $$PWD/src/main.cpp
RESOURCES += $$PWD/src/resources/resources.qrc
INCLUDEPATH += $$PWD/src

!exists($$PWD/src/config/config.pri) {
     error('config.pri file not found in src/config. Please copy src/config/config.pri.template to src/config/config.pri, and then edit src/config/config.pri as needed.')
}
include($$PWD/src/config/config.pri)
include($$PWD/src/acquisition/acquisition.pri)
include($$PWD/src/data/data.pri)
include($$PWD/src/gui/gui.pri)
include($$PWD/src/hardware/hardware.pri)

gpu-cuda {
  include($$PWD/src/modules/cuda/cuda.pri)
}

lif {
  include($$PWD/src/modules/lif/lif.pri)
}

motor {
  include($$PWD/src/modules/motor/motor.pri)
}

OTHER_FILES += README.md doc/Doxyfile doc/Makefile doc/source/conf.py doc/source/index.rst

QMAKE_CXXFLAGS_RELEASE -= -O2
QMAKE_CXXFLAGS_RELEASE += -O3


