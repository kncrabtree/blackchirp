 

SOURCES += \
    $$PWD/ftmwdigitizerconfig.cpp \
    $$PWD/ftmwscope.cpp

HEADERS += \
    $$PWD/ftmwdigitizerconfig.h \
    $$PWD/ftmwscope.h

DEFINES += BC_FTMWSCOPE=$$FTMWSCOPE

equals(FTMWSCOPE,0) {
    HEADERS += $$PWD/virtualftmwscope.h
	SOURCES += $$PWD/virtualftmwscope.cpp
	RESOURCES += $$PWD/../../../resources/virtualdata.qrc
}

equals(FTMWSCOPE,1) {
    HEADERS += $$PWD/dsa71604c.h
	SOURCES += $$PWD/dsa71604c.cpp
}

equals(FTMWSCOPE,2) {
    HEADERS += $$PWD/mso72004c.h
	SOURCES += $$PWD/mso72004c.cpp
}

equals(FTMWSCOPE,3) {
     HEADERS += $$PWD/m4i2220x8.h
	 SOURCES += $$PWD/m4i2220x8.cpp
}

equals(FTMWSCOPE,4) {
     HEADERS += $$PWD/dsox92004a.h
	 SOURCES += $$PWD/dsox92004a.cpp
}

equals(FTMWSCOPE,5) {
     HEADERS += $$PWD/mso64b.h
         SOURCES += $$PWD/mso64b.cpp
}

equals(FTMWSCOPE,6) {
     SOURCES += $$PWD/dsov204a.cpp
	 HEADERS += $$PWD/dsov204a.h
}

equals(FTMWSCOPE,7) {
     SOURCES += $$PWD/dpo71254b.cpp
	 HEADERS += $$PWD/dpo71254b.h
}

equals(FTMWSCOPE,8) {
     SOURCES +=
	 HEADERS +=
}
