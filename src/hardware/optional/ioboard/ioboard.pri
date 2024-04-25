 
HEADERS += \
    $$PWD/ioboard.h \
    $$PWD/ioboardconfig.h

SOURCES += \
    $$PWD/ioboard.cpp \
    $$PWD/ioboardconfig.cpp

NIOB = $$size(IOBOARD)
greaterThan(NIOB,0) {
DEFINES += BC_IOBOARD
DEFINES += BC_NUM_IOBOARD=$$NIOB
for(num, 0..$$NIOB) {
    N = $$upper($$member(IOBOARD,$$num))
	count(N,1) {
	    DEFINES += BC_IOBOARD_$$num=BC_IOBOARD_$$N
		equals(N,VIRTUAL) {
		    DEFINES *= BC_IOBOARD_$$N=VirtualIOBoard
			HEADERS *= $$PWD/virtualioboard.h
			SOURCES *= $$PWD/virtualioboard.cpp
			OPTHW *= "$${H}include <hardware/optional/ioboard/virtualioboard.h>"
		}
		equals(N,LABJACKU3) {
		    DEFINES *= BC_IOBOARD_$$N=LabjackU3
			HEADERS *= $$PWD/labjacku3.h $$PWD/u3.h
			SOURCES *= $$PWD/labjacku3.cpp $$PWD/u3.cpp
			OPTHW *= "$${H}include <hardware/optional/ioboard/labjacku3.h>"
			}
	}
}
}
