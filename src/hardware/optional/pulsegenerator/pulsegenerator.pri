 

HEADERS += \
    $$PWD/pulsegenconfig.h \
    $$PWD/pulsegenerator.h

SOURCES += \
    $$PWD/pulsegenconfig.cpp \
    $$PWD/pulsegenerator.cpp

NPGEN = $$size(PGEN)
greaterThan(NPGEN, 0) {
DEFINES += BC_PGEN
DEFINES += BC_NUM_PGEN=$$NPGEN
for(num, 0..$$NPGEN) {
    N = $$upper($$member(PGEN,$$num))
    count(N,1) {
        DEFINES += BC_PGEN_$$num=BC_PGEN_$$N
        equals(N,VIRTUAL) {
		    DEFINES *= BC_PGEN_$$N=VirtualPulseGenerator
            HEADERS *= $$PWD/virtualpulsegenerator.h
            SOURCES *= $$PWD/virtualpulsegenerator.cpp
            HW *= "$${H}include <hardware/optional/pulsegenerator/virtualpulsegenerator.h>"
        }
        equals(N,QC9528) {
		    DEFINES *= BC_PGEN_$$N=Qc9528
            HEADERS *= $$PWD/qcpulsegenerator.h
            SOURCES *= $$PWD/qc9528.cpp \
            $$PWD/qcpulsegenerator.cpp
            HW *= "$${H}include <hardware/optional/pulsegenerator/qcpulsegenerator.h>"
        }
        equals(N,QC9518) {
		    DEFINES *= BC_PGEN_$$N=Qc9518
            HEADERS *= $$PWD/qcpulsegenerator.h
            SOURCES *= $$PWD/qc9518.cpp \
            $$PWD/qcpulsegenerator.cpp
            HW *= "$${H}include <hardware/optional/pulsegenerator/qcpulsegenerator.h>"
        }
        equals(N,QC9214) {
		    DEFINES *= BC_PGEN_$$N=Qc9214
            HEADERS *= $$PWD/qcpulsegenerator.h
            SOURCES *= $$PWD/qc9214.cpp \
            $$PWD/qcpulsegenerator.cpp
            HW *= "$${H}include <hardware/optional/pulsegenerator/qcpulsegenerator.h>"
        }
        equals(N,DG645) {
            DEFINES *= BC_PGEN_$$N=SRSDG645
            HEADERS *= $$PWD/srsdg645.h
            SOURCES *= $$PWD/srsdg645.cpp
            HW *= "$${H}include <hardware/optional/pulsegenerator/srsdg645.h>"
        }
        equals(N,BNC577_4) {
            DEFINES *= BC_PGEN_$$N=Bnc577_4
            HEADERS *= $$PWD/bnc577.h \
                       $$PWD/qcpulsegenerator.h
            SOURCES *= $$PWD/bnc577.cpp \
                       $$PWD/qcpulsegenerator.cpp
            HW *= "$${H}include <hardware/optional/pulsegenerator/bnc577.h>"
        }
        equals(N,BNC577_8) {
            DEFINES *= BC_PGEN_$$N=Bnc577_8
            HEADERS *= $$PWD/bnc577.h \
                       $$PWD/qcpulsegenerator.h
            SOURCES *= $$PWD/bnc577.cpp \
                       $$PWD/qcpulsegenerator.cpp
            HW *= "$${H}include <hardware/optional/pulsegenerator/bnc577.h>"
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
    DEFINES *= BC_PGEN_BNC577_4
    DEFINES *= BC_PGEN_BNC577_8
    HEADERS *= $$PWD/qcpulsegenerator.h \
    $$PWD/bnc577.h
    SOURCES *= $$PWD/qc9528.cpp \
    $$PWD/bnc577.cpp \
               $$PWD/qcpulsegenerator.cpp \
               $$PWD/qc9518.cpp \
               $$PWD/qc9214.cpp
    HW *= "$${H}include <hardware/optional/pulsegenerator/qcpulsegenerator.h>"
    DEFINES *= BC_PGEN_SRSDG635
    HEADERS *= $$PWD/srsdg645.h
    SOURCES *= $$PWD/srsdg645.cpp
    HW *= "$${H}include <hardware/optional/pulsegenerator/srsdg645.h>"
}
