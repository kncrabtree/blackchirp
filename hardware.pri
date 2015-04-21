include(implementations.pri)

HEADERS += \
    $$PWD/hardwaremanager.h \
    $$PWD/hardwareobject.h \
    $$PWD/tcpinstrument.h \
    $$PWD/rs232instrument.h \
    $$PWD/ftmwscope.h \
    $$PWD/awg.h \
    $$PWD/pulsegenerator.h \
    $$PWD/communicationprotocol.h \
    $$PWD/virtualinstrument.h \
    $$PWD/synthesizer.h \
    $$PWD/flowcontroller.h

SOURCES += \
    $$PWD/hardwaremanager.cpp \
    $$PWD/hardwareobject.cpp \
    $$PWD/tcpinstrument.cpp \
    $$PWD/rs232instrument.cpp \
    $$PWD/ftmwscope.cpp \
    $$PWD/awg.cpp \
    $$PWD/pulsegenerator.cpp \
    $$PWD/communicationprotocol.cpp \
    $$PWD/virtualinstrument.cpp \
    $$PWD/synthesizer.cpp \
    $$PWD/flowcontroller.cpp
