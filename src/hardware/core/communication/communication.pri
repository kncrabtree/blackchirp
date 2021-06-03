

HEADERS += \
    $$PWD/communicationprotocol.h \
    $$PWD/custominstrument.h \
    $$PWD/rs232instrument.h \
    $$PWD/tcpinstrument.h \
    $$PWD/virtualinstrument.h

SOURCES += \
    $$PWD/communicationprotocol.cpp \
    $$PWD/custominstrument.cpp \
    $$PWD/rs232instrument.cpp \
    $$PWD/tcpinstrument.cpp \
    $$PWD/virtualinstrument.cpp

greaterThan(GPIB,-1) {
    HEADERS += \
	           $$PWD/gpibinstrument.h
	SOURCES += \
	           $$PWD/gpibinstrument.cpp
}
