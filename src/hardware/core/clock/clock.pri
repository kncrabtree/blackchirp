HEADERS += \
    $$PWD/clock.h \
    $$PWD/clockmanager.h

SOURCES += \
    $$PWD/clock.cpp \
    $$PWD/clockmanager.cpp


CLOCK_HW = FixedClock Valon5009 Valon5015

for(num, 0..$$size(CLOCKS)) {
N = $$member(CLOCKS,$$num)
count(N,1) {
DEFINES += BC_CLOCK_$$num=$$member(CLOCK_HW,$$N)
}
}

contains(CLOCKS,0) {
    DEFINES += BC_CLOCK_FIXED
	HEADERS += $$PWD/fixedclock.h
	SOURCES += $$PWD/fixedclock.cpp
}
contains(CLOCKS,1) {
    DEFINES += BC_CLOCK_VALON5009
	HEADERS += $$PWD/valon5009.h
	SOURCES += $$PWD/valon5009.cpp
}
contains(CLOCKS,2) {
    DEFINES += BC_CLOCK_VALON5015
	HEADERS += $$PWD/valon5015.h
	SOURCES += $$PWD/valon5015.cpp
}
