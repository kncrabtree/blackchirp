
HEADERS += $$PWD/pressurecontroller.h \
           $$PWD/pressurecontrollerconfig.h
SOURCES += $$PWD/pressurecontroller.cpp \
           $$PWD/pressurecontrollerconfig.cpp


NPC = $$size(PC)
greaterThan(NPC,0) {
DEFINES += BC_PCONTROLLER
DEFINES += BC_NUM_PCONTROLLER=$$NPC
for(num, 0..$$NPC) {
    N = $$upper($$member(PC,$$num))
	count(N,1) {
	    DEFINES += BC_PCONTROLLER_$$num=BC_PCONTROLLER_$$N
		equals(N,VIRTUAL) {
		    DEFINES *= BC_PCONTROLLER_$$N=VirtualPressureController
			HEADERS *= $$PWD/virtualpressurecontroller.h
			SOURCES *= $$PWD/virtualpressurecontroller.cpp
			OPTHW *= "$${H}include <hardware/optional/pressurecontroller/virtualpressurecontroller.h>"
		}
		equals(N,INTELLISYS) {
		    DEFINES *= BC_PCONTROLLER_$$N=IntellisysIQPlus
			HEADERS *= $$PWD/intellisysiqplus.h
			SOURCES *= $$PWD/intellisysiqplus.cpp
			OPTHW *= "$${H}include <hardware/optional/pressurecontroller/intellisysiqplus.h>"
		}
	}
}
}
