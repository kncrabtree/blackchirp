
HEADERS += $$PWD/temperaturecontroller.h \
           $$PWD/temperaturecontrollerconfig.h
SOURCES += $$PWD/temperaturecontroller.cpp \
           $$PWD/temperaturecontrollerconfig.cpp

!lessThan(TC,0) {
    DEFINES += BC_TEMPCONTROLLER=$$TC

    equals(TC,0) {
	    HEADERS += $$PWD/virtualtempcontroller.h
		SOURCES += $$PWD/virtualtempcontroller.cpp
		}
	equals(TC,1) {
        HEADERS += $$PWD/lakeshore218.h
		SOURCES += $$PWD/lakeshore218.cpp
	}
}
