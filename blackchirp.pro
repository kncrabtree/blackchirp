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
