
HEADERS += $$PWD/temperaturecontroller.h \
           $$PWD/temperaturecontrollerconfig.h
SOURCES += $$PWD/temperaturecontroller.cpp \
           $$PWD/temperaturecontrollerconfig.cpp

count(TC,1) {

    N = $$upper($$TC)

    equals(N,VIRTUAL) {
	    DEFINES += BC_TEMPCONTROLLER=VirtualTemperatureController BC_TEMPCONTROLLER_H=virtualtempcontroller.h
	    HEADERS += $$PWD/virtualtempcontroller.h
		SOURCES += $$PWD/virtualtempcontroller.cpp
		}
	equals(N,LAKESHORE218) {
	    DEFINES += BC_TEMPCONTROLLER=Lakeshore218 BC_TEMPCONTROLLER_H=lakeshore218.h
        HEADERS += $$PWD/lakeshore218.h
		SOURCES += $$PWD/lakeshore218.cpp
	}
}
