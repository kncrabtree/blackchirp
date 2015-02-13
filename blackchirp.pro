#-------------------------------------------------
#
# Project created by QtCreator 2015-02-11T14:07:58
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = blackchirp
TEMPLATE = app


SOURCES += main.cpp

include(gui.pri)
include(data.pri)
include(hardware.pri)

