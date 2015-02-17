#-------------------------------------------------
#
# Project created by QtCreator 2015-02-11T14:07:58
#
#-------------------------------------------------

QT       += core gui network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets serialport

TARGET = blackchirp
TEMPLATE = app


SOURCES += main.cpp

include(gui.pri)
include(data.pri)
include(hardware.pri)


unix:!macx: LIBS += -lqwt -lgsl -lm -lgslcblas

#------------------------------------------------
# The following defines allow running the code without real hardware.
# Uncomment the appropriate lines to simulate hardware.
# -----------------------------------------------

# Simulates ALL hardware
#DEFINES += BC_NOHARDWARE

# Simulates ALL RS232 devices
#DEFINES += BC_NORS232

# Simulates ALL TCP devices
#DEFINES += BC_NOTCP

# Simulates FTMW Oscilloscope (uncomment DEFINES and RESOURCES lines)
DEFINES += BC_NOFTSCOPE
RESOURCES += virtualdata.qrc


