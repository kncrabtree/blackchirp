 

HEADERS += \
    $$PWD/pulsegenconfig.h \
    $$PWD/pulsegenerator.h \
    $$PWD/pulsegeneratormanager.h

SOURCES += \
    $$PWD/pulsegenconfig.cpp \
    $$PWD/pulsegenerator.cpp \
    $$PWD/pulsegeneratormanager.cpp

count(PGEN,1) {

    N = $$upper($$PGEN)
	DEFINES += BC_PGEN_$${N}

    equals(N,VIRTUAL) {
	    DEFINES += BC_PGEN=VirtualPulseGenerator BC_PGEN_H=virtualpulsegenerator.h
        HEADERS += $$PWD/virtualpulsegenerator.h
        SOURCES += $$PWD/virtualpulsegenerator.cpp
    }
	equals(N,QC9528) {
	    DEFINES += BC_PGEN=Qc9528 BC_PGEN_H=qcpulsegenerator.h
        HEADERS += $$PWD/qcpulsegenerator.h
        SOURCES += $$PWD/qc9528.cpp \
        $$PWD/qcpulsegenerator.cpp
    }
	equals(N,QC9518) {
	    DEFINES += BC_PGEN=Qc9518 BC_PGEN_H=qcpulsegenerator.h
        HEADERS += $$PWD/qcpulsegenerator.h
        SOURCES += $$PWD/qc9518.cpp \
        $$PWD/qcpulsegenerator.cpp
    }
	equals(N,QC9214) {
	    DEFINES += BC_PGEN=Qc9214 BC_PGEN_H=qcpulsegenerator.h
        HEADERS += $$PWD/qcpulsegenerator.h
        SOURCES += $$PWD/qc9214.cpp \
        $$PWD/qcpulsegenerator.cpp
    }
}
