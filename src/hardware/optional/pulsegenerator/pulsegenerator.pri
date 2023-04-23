 

HEADERS += \
    $$PWD/pulsegenconfig.h \
    $$PWD/pulsegenerator.h

SOURCES += \
    $$PWD/pulsegenconfig.cpp \
    $$PWD/pulsegenerator.cpp

!lessThan(PGEN,0) {
    DEFINES += BC_PGEN=$$PGEN

    equals(PGEN,0) {
        HEADERS += $$PWD/virtualpulsegenerator.h
        SOURCES += $$PWD/virtualpulsegenerator.cpp
    }
    equals(PGEN,1) {
        HEADERS += $$PWD/qcpulsegenerator.h
        SOURCES += $$PWD/qc9528.cpp \
        $$PWD/qcpulsegenerator.cpp
    }
    equals(PGEN,2) {
        HEADERS += $$PWD/qcpulsegenerator.h
        SOURCES += $$PWD/qc9518.cpp \
        $$PWD/qcpulsegenerator.cpp
    }
    equals(PGEN,3) {
        HEADERS += $$PWD/qcpulsegenerator.h
        SOURCES += $$PWD/qc9214.cpp \
        $$PWD/qcpulsegenerator.cpp
    }
}
