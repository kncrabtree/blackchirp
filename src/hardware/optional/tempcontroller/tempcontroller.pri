
HEADERS += $$PWD/temperaturecontroller.h \
           $$PWD/temperaturecontrollerconfig.h
SOURCES += $$PWD/temperaturecontroller.cpp \
           $$PWD/temperaturecontrollerconfig.cpp


NTC = $$size(TC)
greaterThan(NTC,0) {
DEFINES += BC_TEMPCONTROLLER
DEFINES += BC_NUM_TEMPCONTROLLER=$$NTC
for(num, 0..$$NTC) {
    N = $$upper($$member(TC,$$num))
    count(N,1) {
        DEFINES += BC_TEMPCONTROLLER_$$num=BC_TEMPCONTROLLER_$$N
        equals(N,VIRTUAL) {
            DEFINES *= BC_TEMPCONTROLLER_$$N=VirtualTemperatureController
            HEADERS *= $$PWD/virtualtempcontroller.h
            SOURCES *= $$PWD/virtualtempcontroller.cpp
            HW *= "$${H}include <hardware/optional/tempcontroller/virtualtempcontroller.h>"
        }
        equals(N,LAKESHORE218) {
            DEFINES *= BC_TEMPCONTROLLER_$$N=Lakeshore218
            HEADERS *= $$PWD/lakeshore218.h
            SOURCES *= $$PWD/lakeshore218.cpp
            HW *= "$${H}include <hardware/optional/tempcontroller/lakeshore218.h>"
	}
    }
}
}

allhardware {
    HEADERS *= $$PWD/virtualtempcontroller.h
    SOURCES *= $$PWD/virtualtempcontroller.cpp
    HW *= "$${H}include <hardware/optional/tempcontroller/virtualtempcontroller.h>"
    HEADERS *= $$PWD/lakeshore218.h
    SOURCES *= $$PWD/lakeshore218.cpp
    HW *= "$${H}include <hardware/optional/tempcontroller/lakeshore218.h>"
}
