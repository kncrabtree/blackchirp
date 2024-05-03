 

HEADERS += \
    $$PWD/pulsegenconfig.h \
    $$PWD/pulsegenerator.h \
    $$PWD/pulsegeneratormanager.h

SOURCES += \
    $$PWD/pulsegenconfig.cpp \
    $$PWD/pulsegenerator.cpp \
    $$PWD/pulsegeneratormanager.cpp

NPGEN = $$size(PGEN)
greaterThan(NPGEN, 0) {
DEFINES += BC_PGEN
DEFINES += BC_NUM_PGEN=$$NPGEN
for(num, 0..$$NPGEN) {
    N = $$upper($$member(PGEN,$$num))
    count(N,1) {
        DEFINES += BC_PGEN_$$num=BC_PGEN_$$N
        equals(N,VIRTUAL) {
            DEFINES *= BC_PGEN_VIRTUAL=VirtualPulseGenerator
            HEADERS *= $$PWD/virtualpulsegenerator.h
            SOURCES *= $$PWD/virtualpulsegenerator.cpp
            HW *= "$${H}include <hardware/optional/pulsegenerator/virtualpulsegenerator.h>"
        }
        equals(N,QC9528) {
            DEFINES *= BC_PGEN_QC9528=Qc9528
            HEADERS *= $$PWD/qcpulsegenerator.h
            SOURCES *= $$PWD/qc9528.cpp \
            $$PWD/qcpulsegenerator.cpp
            HW *= "$${H}include <hardware/optional/pulsegenerator/qcpulsegenerator.h>"
        }
        equals(N,QC9518) {
            DEFINES *= BC_PGEN_QC9518=Qc9518
            HEADERS *= $$PWD/qcpulsegenerator.h
            SOURCES *= $$PWD/qc9518.cpp \
            $$PWD/qcpulsegenerator.cpp
            HW *= "$${H}include <hardware/optional/pulsegenerator/qcpulsegenerator.h>"
        }
        equals(N,QC9214) {
            DEFINES *= BC_PGEN_QC9214=Qc9214
            HEADERS *= $$PWD/qcpulsegenerator.h
            SOURCES *= $$PWD/qc9214.cpp \
            $$PWD/qcpulsegenerator.cpp
            HW *= "$${H}include <hardware/optional/pulsegenerator/qcpulsegenerator.h>"
        }
    }
}
}

allhardware {
    HEADERS *= $$PWD/virtualpulsegenerator.h
    SOURCES *= $$PWD/virtualpulsegenerator.cpp
    HW *= "$${H}include <hardware/optional/pulsegenerator/virtualpulsegenerator.h>"
    DEFINES *= BC_PGEN_QC9528
    DEFINES *= BC_PGEN_QC9518
    DEFINES *= BC_PGEN_QC9214
    HEADERS *= $$PWD/qcpulsegenerator.h
    SOURCES *= $$PWD/qc9528.cpp \
               $$PWD/qcpulsegenerator.cpp \
               $$PWD/qc9518.cpp \
               $$PWD/qc9214.cpp
    HW *= "$${H}include <hardware/optional/pulsegenerator/qcpulsegenerator.h>"
}
