HEADERS += \
    $$PWD/clock.h \
    $$PWD/clockmanager.h

SOURCES += \
    $$PWD/clock.cpp \
    $$PWD/clockmanager.cpp


contains(CLOCKS,0) {
        HEADERS += $$PWD/fixedclock.h
		SOURCES += $$PWD/fixedclock.cpp
}
contains(CLOCKS,1) {
        HEADERS += $$PWD/valon5009.h
		SOURCES += $$PWD/valon5009.cpp
}
contains(CLOCKS,2) {
        HEADERS += $$PWD/valon5015.h
		SOURCES += $$PWD/valon5015.cpp
}
