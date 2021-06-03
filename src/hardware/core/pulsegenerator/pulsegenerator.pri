 

HEADERS += \
    $$PWD/pulsegenerator.h

SOURCES += \
    $$PWD/pulsegenerator.cpp


equals(PGEN,0) {
    HEADERS += $$PWD/virtualpulsegenerator.h
	SOURCES += $$PWD/virtualpulsegenerator.cpp
}
equals(PGEN,1) {
    HEADERS += $$PWD/qc9528.h
	SOURCES += $$PWD/qc9528.cpp
}
equals(PGEN,2) {
    HEADERS += $$PWD/qc9518.h
	SOURCES += $$PWD/qc9518.cpp
}
