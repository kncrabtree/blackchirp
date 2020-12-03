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
    $$PWD/flowcontroller.h \
    $$PWD/ioboard.h \
		$$PWD/custominstrument.h \
    $$PWD/clockmanager.h \
    $$PWD/clock.h

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
    $$PWD/flowcontroller.cpp \
    $$PWD/ioboard.cpp \
		$$PWD/custominstrument.cpp \
    $$PWD/clockmanager.cpp \
    $$PWD/clock.cpp

allhardware {
HEADERS += \
	$$PWD/lifscope.h \
	$$PWD/gpibcontroller.h \
	$$PWD/gpibinstrument.h \
    $$PWD/motorcontroller.h \
    $$PWD/motoroscilloscope.h \
    $$PWD/pressurecontroller.h \
    $$PWD/temperaturecontroller.h

SOURCES += \
	$$PWD/lifscope.cpp \
	$$PWD/gpibcontroller.cpp \
	$$PWD/gpibinstrument.cpp \
    $$PWD/motorcontroller.cpp \
    $$PWD/motoroscilloscope.cpp \
    $$PWD/pressurecontroller.cpp \
    $$PWD/temperaturecontroller.cpp
} else {

greaterThan(GPIB,-1) {
	HEADERS += $$PWD/gpibcontroller.h \
			   $$PWD/gpibinstrument.h
	SOURCES += $$PWD/gpibcontroller.cpp \
			   $$PWD/gpibinstrument.cpp
}

greaterThan(PC,-1) {
	HEADERS += $$PWD/pressurecontroller.h
	SOURCES += $$PWD/pressurecontroller.cpp
}

greaterThan(TC,-1) {
     HEADERS += $$PWD/temperaturecontroller.h
     SOURCES += $$PWD/temperaturecontroller.cpp
}

lif {
	HEADERS += $$PWD/lifscope.h
	SOURCES += $$PWD/lifscope.cpp
}

motor {
        HEADERS += $$PWD/motorcontroller.h \
                   $$PWD/motoroscilloscope.h
        SOURCES += $$PWD/motorcontroller.cpp \
                   $$PWD/motoroscilloscope.cpp
}

}
