 

HEADERS += \
    $$PWD/lifdigitizerconfig.h \
    $$PWD/lifscope.h

SOURCES += \
    $$PWD/lifdigitizerconfig.cpp \
    $$PWD/lifscope.cpp

N = $$upper($$LIFSCOPE)

equals(N,VIRTUAL) {
    DEFINES += BC_LIFSCOPE=VirtualLifScope
    HEADERS += $$PWD/virtuallifscope.h
    SOURCES += $$PWD/virtuallifscope.cpp
    LIFHW *= "$${H}include <modules/lif/hardware/lifdigitizer/virtuallifscope.h>"
}
equals(N,M4I2211X8) {
    DEFINES += BC_LIFSCOPE=M4i2211x8
    HEADERS +=  $$PWD/m4i2211x8.h
    SOURCES +=  $$PWD/m4i2211x8.cpp
    LIFHW *= "$${H}include <modules/lif/hardware/lifdigitizer/m4i2211x8.h>"
}

allhardware {
    HEADERS *= $$PWD/virtuallifscope.h
    SOURCES *= $$PWD/virtuallifscope.cpp
    LIFHW *= "$${H}include <modules/lif/hardware/lifdigitizer/virtuallifscope.h>"
    HEADERS *=  $$PWD/m4i2211x8.h
    SOURCES *=  $$PWD/m4i2211x8.cpp
    LIFHW *= "$${H}include <modules/lif/hardware/lifdigitizer/m4i2211x8.h>"
}
