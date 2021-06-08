QT += testlib
QT += gui
CONFIG += qt warn_on depend_includepath testcase c++17

TEMPLATE = app

SOURCES +=  tests/tst_settingsstoragetest.cpp \
    src/data/storage/settingsstorage.cpp

INCLUDEPATH += src tests

HEADERS += \
    src/data/storage/settingsstorage.h
