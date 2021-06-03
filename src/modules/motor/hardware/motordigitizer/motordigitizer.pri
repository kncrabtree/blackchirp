 

HEADERS += \
    $$PWD/motoroscilloscope.h

SOURCES += \
    $$PWD/motoroscilloscope.cpp


equals(MOTORSCOPE,0) {
        HEADERS += $$PWD/virtualmotorscope.h
		SOURCES += $$PWD/virtualmotorscope.cpp
}
equals(MOTORSCOPE,1) {
        HEADERS += $$PWD/pico2206b.h
		SOURCES += $$PWD/pico2206b.cpp
}
