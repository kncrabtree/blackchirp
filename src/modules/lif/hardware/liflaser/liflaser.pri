 

HEADERS += \
    $$PWD/liflaser.h

SOURCES += \
    $$PWD/liflaser.cpp

N = $$upper($$LIFLASER)

equals(N,VIRTUAL) {
    DEFINES += BC_LIFLASER=VirtualLifLaser BC_LIFLASER_H=virtualliflaser.h
    HEADERS += $$PWD/virtualliflaser.h
	SOURCES += $$PWD/virtualliflaser.cpp
}

equals(N,OPOLETTE) {
    DEFINES += BC_LIFLASER=Opolette BC_LIFLASER_H=opolette.h
    HEADERS += $$PWD/opolette.h
        SOURCES += $$PWD/opolette.cpp
}
