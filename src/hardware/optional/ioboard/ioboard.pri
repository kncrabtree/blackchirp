 
HEADERS += \
    $$PWD/ioboard.h \
    $$PWD/ioboardconfig.h

SOURCES += \
    $$PWD/ioboard.cpp \
    $$PWD/ioboardconfig.cpp

count(IOBOARD,1) {
    N = $$upper($$IOBOARD)

    equals(N,VIRTUAL) {
	    DEFINES += BC_IOBOARD=VirtualIOBoard BC_IOBOARD_H=virtualioboard.h
	    HEADERS += $$PWD/virtualioboard.h
		SOURCES += $$PWD/virtualioboard.cpp
	}
	equals(N,LABJACKU3) {
	    DEFINES += BC_IOBOARD=LabjackU3 BC_IOBOARD_H=labjacku3.h
	    HEADERS += $$PWD/labjacku3.h \
		           $$PWD/u3.h
		SOURCES += $$PWD/labjacku3.cpp \
		           $$PWD/u3.cpp
	}
}
