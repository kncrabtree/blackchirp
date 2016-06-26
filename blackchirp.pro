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

CONFIG += c++11

SOURCES += main.cpp

RESOURCES += resources.qrc

unix:!macx: LIBS += -lqwt -lgsl -lm -lgslcblas

include(config.pri)
include(acquisition.pri)
include(gui.pri)
include(data.pri)
include(hardware.pri)
include(wizard.pri)
include(implementations.pri)

QMAKE_CXXFLAGS_RELEASE -= -O2
QMAKE_CXXFLAGS_RELEASE += -O3

DISTFILES += \
    52-serial.rules \
    config.pri.template \
    COPYING \
    COPYING.qwt \
    COPYING.lesser \
    README


