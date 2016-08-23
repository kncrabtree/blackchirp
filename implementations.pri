HEADERS += \
    $$PWD/virtualftmwscope.h \
    $$PWD/dsa71604c.h \
    $$PWD/virtualawg.h \
    $$PWD/virtualvalonsynth.h \
    $$PWD/virtualpulsegenerator.h \
    $$PWD/virtualflowcontroller.h \
    $$PWD/virtuallifscope.h \
    $$PWD/virtualgpibcontroller.h \
    $$PWD/awg70002a.h \
    $$PWD/qc9528.h \
    $$PWD/valon5009.h \
    $$PWD/mks647c.h \
    $$PWD/prologixgpiblan.h \
    $$PWD/virtualioboard.h \
    $$PWD/mso72004c.h \
    $$PWD/qc9518.h \
    $$PWD/pldrogroup.h \
    $$PWD/awg7122b.h \
    $$PWD/virtualmotorcontroller.h \
    $$PWD/scx11.h


SOURCES += \
    $$PWD/virtualftmwscope.cpp \
    $$PWD/dsa71604c.cpp \
    $$PWD/virtualawg.cpp \
    $$PWD/virtualvalonsynth.cpp \
    $$PWD/virtualpulsegenerator.cpp \
    $$PWD/virtualflowcontroller.cpp \
    $$PWD/virtuallifscope.cpp \
    $$PWD/virtualgpibcontroller.cpp \
    $$PWD/awg70002a.cpp \
    $$PWD/qc9528.cpp \
    $$PWD/valon5009.cpp \
    $$PWD/mks647c.cpp \
    $$PWD/prologixgpiblan.cpp \
    $$PWD/virtualioboard.cpp \
    $$PWD/mso72004c.cpp \
    $$PWD/qc9518.cpp \
    $$PWD/pldrogroup.cpp \
    $$PWD/awg7122b.cpp \
    $$PWD/virtualmotorcontroller.cpp \
    $$PWD/scx11.cpp


equals(IOBOARD,1) {
HEADERS += $$PWD/labjacku3.h \
    $$PWD/u3.h

SOURCES += $$PWD/labjacku3.cpp \
    $$PWD/u3.cpp

LIBS += -llabjackusb
}
