 
HEADERS += $$PWD/gpibcontroller.h
SOURCES += $$PWD/gpibcontroller.cpp

!lessThan(GPIB,0) {

    DEFINES += BC_GPIBCONTROLLER=$$GPIB
	equals(GPIB,0) {
	    HEADERS += $$PWD/virtualgpibcontroller.h
		SOURCES += $$PWD/virtualgpibcontroller.cpp
		}
	equals(GPIB,1) {
	    HEADERS += $$PWD/prologixgpiblan.h
		SOURCES += $$PWD/prologixgpiblan.cpp
	}
        equals(GPIB,2) {
            HEADERS += $$PWD/prologixgpibusb.h
                SOURCES += $$PWD/prologixgpibusb.cpp
        }
}
