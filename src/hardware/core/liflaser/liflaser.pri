 

HEADERS += \
    $$PWD/liflaser.h

SOURCES += \
    $$PWD/liflaser.cpp

N = $$upper($$LIFLASER)

equals(N,VIRTUAL) {
    DEFINES += BC_LIFLASER=VirtualLifLaser
    HEADERS += $$PWD/virtualliflaser.h
    SOURCES += $$PWD/virtualliflaser.cpp
    LIFHW *= "$${H}include <modules/lif/hardware/liflaser/virtualliflaser.h>"
}

equals(N,OPOLETTE) {
    DEFINES += BC_LIFLASER=Opolette
    HEADERS += $$PWD/opolette.h
    SOURCES += $$PWD/opolette.cpp
    LIFHW *= "$${H}include <modules/lif/hardware/liflaser/opolette.h>"
}

equals(N,SIRAHCOBRA) {
    DEFINES += BC_LIFLASER=SirahCobra
    HEADERS += $$PWD/sirahcobra.h
    SOURCES += $$PWD/sirahcobra.cpp
    LIFHW *= "$${H}include <modules/lif/hardware/liflaser/sirahcobra.h>"
}

allhardware {
    HEADERS *= $$PWD/virtualliflaser.h
    SOURCES *= $$PWD/virtualliflaser.cpp
    LIFHW *= "$${H}include <modules/lif/hardware/liflaser/virtualliflaser.h>"
    HEADERS *= $$PWD/opolette.h
    SOURCES *= $$PWD/opolette.cpp
    LIFHW *= "$${H}include <modules/lif/hardware/liflaser/opolette.h>"
    HEADERS *= $$PWD/sirahcobra.h
    SOURCES *= $$PWD/sirahcobra.cpp
    LIFHW *= "$${H}include <modules/lif/hardware/liflaser/sirahcobra.h>"
}
