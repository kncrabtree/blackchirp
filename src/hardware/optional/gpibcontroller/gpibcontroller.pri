 
HEADERS += $$PWD/gpibcontroller.h
SOURCES += $$PWD/gpibcontroller.cpp

count(GPIB,1) {

    N = $$upper($$GPIB)
	equals(N,VIRTUAL) {
            DEFINES += BC_GPIBCONTROLLER=VirtualGpibController
	    HEADERS += $$PWD/virtualgpibcontroller.h
            SOURCES += $$PWD/virtualgpibcontroller.cpp
            HW *= "$${H}include <hardware/optional/gpibcontroller/virtualgpibcontroller.h>"
        }
	equals(N,PROLOGIXLAN) {
            DEFINES += BC_GPIBCONTROLLER=PrologixGpibLan
	    HEADERS += $$PWD/prologixgpiblan.h
            SOURCES += $$PWD/prologixgpiblan.cpp
            HW *= "$${H}include <hardware/optional/chirpsource/prologizgpiblan.h>"
	}
	equals(N,PROLOGIXUSB) {
            DEFINES += BC_GPIBCONTROLLER=PrologixGpibUsb
            HEADERS += $$PWD/prologixgpibusb.h
            SOURCES += $$PWD/prologixgpibusb.cpp
            HW *= "$${H}include <hardware/optional/chirpsource/prologixgpibusb.h>"
        }
}

allhardware {
    HEADERS *= $$PWD/virtualgpibcontroller.h
    SOURCES *= $$PWD/virtualgpibcontroller.cpp
    HW *= "$${H}include <hardware/optional/gpibcontroller/virtualgpibcontroller.h>"
    HEADERS *= $$PWD/prologixgpiblan.h
    SOURCES *= $$PWD/prologixgpiblan.cpp
    HW *= "$${H}include <hardware/optional/chirpsource/prologizgpiblan.h>"
    HEADERS *= $$PWD/prologixgpibusb.h
    SOURCES *= $$PWD/prologixgpibusb.cpp
    HW *= "$${H}include <hardware/optional/chirpsource/prologixgpibusb.h>"
}
