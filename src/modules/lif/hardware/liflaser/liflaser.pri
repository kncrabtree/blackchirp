 

HEADERS += \
    $$PWD/liflaser.h

SOURCES += \
    $$PWD/liflaser.cpp


equals(LIFLASER,0) {
    HEADERS += $$PWD/virtualliflaser.h
	SOURCES += $$PWD/virtualliflaser.cpp
}
