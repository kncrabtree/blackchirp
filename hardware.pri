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
    $$PWD/flowcontroller.h \
    $$PWD/ioboard.h \
	$$PWD/custominstrument.h

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
    $$PWD/flowcontroller.cpp \
    $$PWD/ioboard.cpp \
	$$PWD/custominstrument.cpp

allhardware {
HEADERS += \
	$$PWD/lifscope.h \
	$$PWD/gpibcontroller.h \
	$$PWD/gpibinstrument.h \
	$$PWD/motorcontroller.h

SOURCES += \
	$$PWD/lifscope.cpp \
	$$PWD/gpibcontroller.cpp \
	$$PWD/gpibinstrument.cpp \
	$$PWD/motorcontroller.cpp
} else {

greaterThan(GPIB,-1) {
	HEADERS += $$PWD/gpibcontroller.h \
			   $$PWD/gpibinstrument.h
	SOURCES += $$PWD/gpibcontroller.cpp \
			   $$PWD/gpibinstrument.cpp
}

lif {
	HEADERS += $$PWD/lifscope.h
	SOURCES += $$PWD/lifscope.cpp
}

motor {
	HEADERS += $$PWD/motorcontroller.h
	SOURCES += $$PWD/motorcontroller.cpp
}

}
