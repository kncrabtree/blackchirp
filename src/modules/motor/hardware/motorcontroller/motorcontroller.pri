 

HEADERS += \
    $$PWD/motorcontroller.h

SOURCES += \
    $$PWD/motorcontroller.cpp

equals(MOTOR,0) {
    HEADERS += $$PWD/virtualmotorcontroller.h
	SOURCES += $$PWD/virtualmotorcontroller.cpp
}
equals(MOTOR,1) {
    HEADERS += $$PWD/scx11.h
	SOURCES += $$PWD/scx11.cpp
}
