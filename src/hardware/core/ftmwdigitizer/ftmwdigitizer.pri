 

SOURCES += \
    $$PWD/ftmwdigitizerconfig.cpp \
    $$PWD/ftmwscope.cpp

HEADERS += \
    $$PWD/ftmwdigitizerconfig.h \
    $$PWD/ftmwscope.h

N = $$upper($$FTMWSCOPE)

equals(N,VIRTUAL) {
    DEFINES += BC_FTMWSCOPE=VirtualFtmwScope BC_FTMWSCOPE_H=virtualftmwscope.h
    HEADERS += $$PWD/virtualftmwscope.h
	SOURCES += $$PWD/virtualftmwscope.cpp
	RESOURCES += $$PWD/../../../resources/virtualdata.qrc
}

equals(N,DSA71604C) {
    DEFINES += BC_FTMWSCOPE=Dsa71604c BC_FTMWSCOPE_H=dsa71604c.h
    HEADERS += $$PWD/dsa71604c.h
	SOURCES += $$PWD/dsa71604c.cpp
}

equals(N,MSO72004C) {
    DEFINES += BC_FTMWSCOPE=MSO72004C BC_FTMWSCOPE_H=mso72004c.h
    HEADERS += $$PWD/mso72004c.h
	SOURCES += $$PWD/mso72004c.cpp
}

equals(N,M4I2220X8) {
    DEFINES += BC_FTMWSCOPE=M4i2220x8 BC_FTMWSCOPE_H=m4i2220x8.h
	HEADERS += $$PWD/m4i2220x8.h
	SOURCES += $$PWD/m4i2220x8.cpp
}

equals(N,DSOX92004A) {
    DEFINES += BC_FTMWSCOPE=DSOx92004A BC_FTMWSCOPE_H=dsox92004a.h
	HEADERS += $$PWD/dsox92004a.h
	SOURCES += $$PWD/dsox92004a.cpp
}

equals(N,MSO64B) {
    DEFINES += BC_FTMWSCOPE=MSO64B BC_FTMWSCOPE_H=mso64b.h
	HEADERS += $$PWD/mso64b.h
	SOURCES += $$PWD/mso64b.cpp
}
