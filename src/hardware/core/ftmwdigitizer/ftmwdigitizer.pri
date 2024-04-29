 

SOURCES += \
    $$PWD/ftmwdigitizerconfig.cpp \
    $$PWD/ftmwscope.cpp

HEADERS += \
    $$PWD/ftmwdigitizerconfig.h \
    $$PWD/ftmwscope.h

N = $$upper($$FTMWSCOPE)

equals(N,VIRTUAL) {
    DEFINES += BC_FTMWSCOPE=VirtualFtmwScope
    HEADERS += $$PWD/virtualftmwscope.h
	SOURCES += $$PWD/virtualftmwscope.cpp
	RESOURCES += $$PWD/../../../resources/virtualdata.qrc
    HW *= "$${H}include <hardware/core/ftmwdigitizer/virtualftmwscope.h>"
}

equals(N,DSA71604C) {
    DEFINES += BC_FTMWSCOPE=Dsa71604c
    HEADERS += $$PWD/dsa71604c.h
	SOURCES += $$PWD/dsa71604c.cpp
    HW *= "$${H}include <hardware/core/ftmwdigitizer/dsa71604c.h>"
}

equals(N,MSO72004C) {
    DEFINES += BC_FTMWSCOPE=MSO72004C
    HEADERS += $$PWD/mso72004c.h
	SOURCES += $$PWD/mso72004c.cpp
    HW *= "$${H}include <hardware/core/ftmwdigitizer/mso72004c.h>"
}

equals(N,M4I2220X8) {
    DEFINES += BC_FTMWSCOPE=M4i2220x8
	HEADERS += $$PWD/m4i2220x8.h
	SOURCES += $$PWD/m4i2220x8.cpp
        HW *= "$${H}include <hardware/core/ftmwdigitizer/m412220x8.h>"
}

equals(N,DSOX92004A) {
    DEFINES += BC_FTMWSCOPE=DSOx92004A
	HEADERS += $$PWD/dsox92004a.h
	SOURCES += $$PWD/dsox92004a.cpp
    HW *= "$${H}include <hardware/core/ftmwdigitizer/dsox92004a.h>"
}

equals(N,MSO64B) {
    DEFINES += BC_FTMWSCOPE=MSO64B
	HEADERS += $$PWD/mso64b.h
	SOURCES += $$PWD/mso64b.cpp
    HW *= "$${H}include <hardware/core/ftmwdigitizer/ms064b.h>"
}

equals(N,DSOV204A) {
    DEFINES += BC_FTMWSCOPE=DSOv204A
     SOURCES += $$PWD/dsov204a.cpp
	 HEADERS += $$PWD/dsov204a.h
    HW *= "$${H}include <hardware/core/ftmwdigitizer/dso204a.h>"
}

allhardware {
    HEADERS *= $$PWD/virtualftmwscope.h
    SOURCES *= $$PWD/virtualftmwscope.cpp
    RESOURCES *= $$PWD/../../../resources/virtualdata.qrc
    HW *= "$${H}include <hardware/core/ftmwdigitizer/virtualftmwscope.h>"
    HEADERS *= $$PWD/dsa71604c.h
    SOURCES *= $$PWD/dsa71604c.cpp
    HW *= "$${H}include <hardware/core/ftmwdigitizer/dsa71604c.h>"
    HEADERS *= $$PWD/mso72004c.h
    SOURCES *= $$PWD/mso72004c.cpp
    HW *= "$${H}include <hardware/core/ftmwdigitizer/mso72004c.h>"
    HEADERS *= $$PWD/m4i2220x8.h
    SOURCES *= $$PWD/m4i2220x8.cpp
    HW *= "$${H}include <hardware/core/ftmwdigitizer/m412220x8.h>"
    HEADERS *= $$PWD/dsox92004a.h
    SOURCES *= $$PWD/dsox92004a.cpp
    HW *= "$${H}include <hardware/core/ftmwdigitizer/dsox92004a.h>"
    HEADERS *= $$PWD/mso64b.h
    SOURCES *= $$PWD/mso64b.cpp
    HW *= "$${H}include <hardware/core/ftmwdigitizer/ms064b.h>"
    SOURCES *= $$PWD/dsov204a.cpp
    HEADERS *= $$PWD/dsov204a.h
    HW *= "$${H}include <hardware/core/ftmwdigitizer/dso204a.h>"
}
