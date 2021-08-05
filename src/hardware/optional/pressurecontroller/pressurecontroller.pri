
greaterThan(PC,-1) {
    HEADERS += $$PWD/pressurecontroller.h
	SOURCES += $$PWD/pressurecontroller.cpp
}


equals(PC,0) {
        HEADERS += $$PWD/virtualpressurecontroller.h
		SOURCES += $$PWD/virtualpressurecontroller.cpp
}
equals(PC,1) {
        HEADERS += $$PWD/intellisysiqplus.h
		SOURCES += $$PWD/intellisysiqplus.cpp
}

HEADERS += \
    $$PWD/pressurecontrollerconfig.h

SOURCES += \
    $$PWD/pressurecontrollerconfig.cpp
