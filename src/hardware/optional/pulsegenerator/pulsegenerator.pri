 

HEADERS += \
    $$PWD/pulsegenconfig.h \
    $$PWD/pulsegenerator.h \
    $$PWD/pulsegeneratormanager.h

SOURCES += \
    $$PWD/pulsegenconfig.cpp \
    $$PWD/pulsegenerator.cpp \
    $$PWD/pulsegeneratormanager.cpp

H = $$LITERAL_HASH

SS = $$size(PGEN)
greaterThan(SS, 0) {
DEFINES += BC_PGEN
DEFINES += BC_NUM_PGEN=$$size(PGEN)
for(num, 0..$$size(PGEN)) {
    N = $$upper($$member(PGEN,$$num))
	count(N,1) {
	    DEFINES += BC_PGEN_$$num=BC_PGEN_$$N

        equals(N,VIRTUAL) {
		    DEFINES *= BC_PGEN_VIRTUAL=VirtualPulseGenerator
			HEADERS *= $$PWD/virtualpulsegenerator.h
			SOURCES *= $$PWD/virtualpulsegenerator.cpp
			OPTHW *= "$${H}include <hardware/optional/pulsegenerator/virtualpulsegenerator.h>"
			}
		equals(N,QC9528) {
		    DEFINES *= BC_PGEN_QC9528=Qc9528
			HEADERS *= $$PWD/qcpulsegenerator.h
			SOURCES *= $$PWD/qc9528.cpp \
			$$PWD/qcpulsegenerator.cpp
			OPTHW *= "$${H}include <hardware/optional/pulsegenerator/qcpulsegenerator.h>"
		}
		equals(N,QC9518) {
		    DEFINES *= BC_PGEN_QC9518=Qc9518
			HEADERS *= $$PWD/qcpulsegenerator.h
			SOURCES *= $$PWD/qc9518.cpp \
			$$PWD/qcpulsegenerator.cpp
			OPTHW *= "$${H}include <hardware/optional/pulsegenerator/qcpulsegenerator.h>"
		}
		equals(N,QC9214) {
		    DEFINES *= BC_PGEN_QC9514=Qc9514
			HEADERS *= $$PWD/qcpulsegenerator.h
			SOURCES *= $$PWD/qc9514.cpp \
			$$PWD/qcpulsegenerator.cpp
			OPTHW *= "$${H}include <hardware/optional/pulsegenerator/qcpulsegenerator.h>"
		}
	}
}
}
