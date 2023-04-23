 
HEADERS += \
    $$PWD/ioboard.h \
    $$PWD/ioboardconfig.h

SOURCES += \
    $$PWD/ioboard.cpp \
    $$PWD/ioboardconfig.cpp

!lessThan(IOBOARD,0) {
    DEFINES += BC_IOBOARD=$$IOBOARD
	equals(IOBOARD,0) {
	    HEADERS += $$PWD/virtualioboard.h
		SOURCES += $$PWD/virtualioboard.cpp
	}
	equals(IOBOARD,1) {
	    HEADERS += $$PWD/labjacku3.h \
		           $$PWD/u3.h
				SOURCES += $$PWD/labjacku3.cpp \
		           $$PWD/u3.cpp
				}
}
