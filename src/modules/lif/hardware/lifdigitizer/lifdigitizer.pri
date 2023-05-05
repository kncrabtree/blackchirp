 

HEADERS += \
    $$PWD/lifdigitizerconfig.h \
    $$PWD/lifscope.h

SOURCES += \
    $$PWD/lifdigitizerconfig.cpp \
    $$PWD/lifscope.cpp

N = $$upper($$LIFSCOPE)

equals(N,VIRTUAL) {
    DEFINES += BC_LIFSCOPE=VirtualLifScope BC_LIFSCOPE_H=virtuallifscope.h
    HEADERS += $$PWD/virtuallifscope.h
	SOURCES += $$PWD/virtuallifscope.cpp
}
equals(N,M4I2211X8) {
    DEFINES += BC_LIFSCOPE=M4i2211x8 BC_LIFSCOPE_H=m4i2211x8.h
    HEADERS +=  $$PWD/m4i2211x8.h
	SOURCES +=  $$PWD/m4i2211x8.cpp
}
