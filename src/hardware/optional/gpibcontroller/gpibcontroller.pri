 
HEADERS += $$PWD/gpibcontroller.h
SOURCES += $$PWD/gpibcontroller.cpp

count(GPIB,1) {

    N = $$upper($$GPIB)
	equals(N,VIRTUAL) {
	    DEFINES += BC_GPIBCONTROLLER=VirtualGpibController BC_GPIBCONTROLLER_H=virtualgpibcontroller.h
	    HEADERS += $$PWD/virtualgpibcontroller.h
		SOURCES += $$PWD/virtualgpibcontroller.cpp
		}
	equals(N,PROLOGIXLAN) {
	    DEFINES += BC_GPIBCONTROLLER=PrologixGpibLan BC_GPIBCONTROLLER_H=prologixgpiblan.h
	    HEADERS += $$PWD/prologixgpiblan.h
		SOURCES += $$PWD/prologixgpiblan.cpp
	}
	equals(N,PROLOGIXUSB) {
	    DEFINES += BC_GPIBCONTROLLER=PrologixGpibUsb BC_GPIBCONTROLLER_H=prologixgpibusb.h
		HEADERS += $$PWD/prologixgpibusb.h
		SOURCES += $$PWD/prologixgpibusb.cpp
		}
}
