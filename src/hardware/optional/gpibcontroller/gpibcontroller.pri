 
greaterThan(GPIB,-1) {
    HEADERS += $$PWD/gpibcontroller.h
	SOURCES += $$PWD/gpibcontroller.cpp
}

equals(GPIB,0) {
    HEADERS += $$PWD/virtualgpibcontroller.h
	SOURCES += $$PWD/virtualgpibcontroller.cpp
}
equals(GPIB,1) {
    HEADERS += $$PWD/prologixgpiblan.h
	SOURCES += $$PWD/prologixgpiblan.cpp
}
