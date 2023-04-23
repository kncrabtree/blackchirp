
HEADERS += $$PWD/pressurecontroller.h \
           $$PWD/pressurecontrollerconfig.h
SOURCES += $$PWD/pressurecontroller.cpp \
           $$PWD/pressurecontrollerconfig.cpp


!lessThan(PC,0) {
    DEFINES += BC_PCONTROLLER=$$PC

    equals(PC,0) {
        HEADERS += $$PWD/virtualpressurecontroller.h
		SOURCES += $$PWD/virtualpressurecontroller.cpp
		}
	equals(PC,1) {
        HEADERS += $$PWD/intellisysiqplus.h
		SOURCES += $$PWD/intellisysiqplus.cpp
		}
}
