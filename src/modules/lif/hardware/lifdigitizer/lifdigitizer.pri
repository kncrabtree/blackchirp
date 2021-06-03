 

HEADERS += \
    $$PWD/lifscope.h

SOURCES += \
    $$PWD/lifscope.cpp

equals(LIFSCOPE,0) {
    HEADERS += $$PWD/virtuallifscope.h
	SOURCES += $$PWD/virtuallifscope.cpp
}
equals(LIFSCOPE,1) {
    HEADERS +=  $$PWD/m4i2211x8.h
	SOURCES +=  $$PWD/m4i2211x8.cpp
}
