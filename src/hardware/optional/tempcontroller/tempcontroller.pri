
greaterThan(TC,-1) {
     HEADERS += $$PWD/temperaturecontroller.h
	 SOURCES += $$PWD/temperaturecontroller.cpp
}

equals(TC,0) {
        HEADERS += $$PWD/virtualtempcontroller.h
		SOURCES += $$PWD/virtualtempcontroller.cpp
}
equals(TC,1) {
        HEADERS += $$PWD/lakeshore218.h
		SOURCES += $$PWD/lakeshore218.cpp
}

HEADERS += \
    $$PWD/temperaturecontrollerconfig.h

SOURCES += \
    $$PWD/temperaturecontrollerconfig.cpp
