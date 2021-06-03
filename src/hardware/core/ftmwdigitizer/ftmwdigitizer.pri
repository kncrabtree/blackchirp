 

SOURCES += \
    $$PWD/ftmwscope.cpp

HEADERS += \
    $$PWD/ftmwscope.h

equals(FTMWSCOPE,0) {
    HEADERS += $$PWD/virtualftmwscope.h
	SOURCES += $$PWD/virtualftmwscope.cpp
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
