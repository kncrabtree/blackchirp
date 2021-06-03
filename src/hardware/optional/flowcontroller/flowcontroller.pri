
greaterThan(FC,-1) {
    HEADERS += $$PWD/flowcontroller.h
	SOURCES += $$PWD/flowcontroller.cpp
}

equals(FC,0) {
    HEADERS += $$PWD/virtualflowcontroller.h
	SOURCES += $$PWD/virtualflowcontroller.cpp
}
equals(FC,1) {
    HEADERS += $$PWD/mks647c.h
	SOURCES += $$PWD/mks647c.cpp
}
equals(FC,2) {
    HEADERS += $$PWD/mks946.h
	SOURCES += $$PWD/mks946.cpp
}
