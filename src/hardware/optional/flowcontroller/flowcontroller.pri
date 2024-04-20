
HEADERS += $$PWD/flowcontroller.h \
           $$PWD/flowconfig.h
SOURCES += $$PWD/flowcontroller.cpp \
           $$PWD/flowconfig.cpp


NFC = $$size(FC)
greaterThan(NFC,0) {
DEFINES += BC_FLOWCONTROLLER
DEFINES += BC_NUM_FLOWCONTROLLER=$$NFC
for(num, 0..$$NFC) {
    N = $$upper($$member(FC,$$num))
	count(N,1) {
	    DEFINES += BC_FLOWCONTROLLER_$$num=BC_FLOWCONTROLLER_$$N
		equals(N,VIRTUAL) {
		    DEFINES *= BC_FLOWCONTROLLER_VIRTUAL=VirtualFlowController
			HEADERS *= $$PWD/virtualflowcontroller.h
			SOURCES *= $$PWD/virtualflowcontroller.cpp
			OPTHW *= "$${H}include <hardware/optional/flowcontroller/virtualflowcontroller.h>"
		}
		equals(N,MKS647C) {
		    DEFINES *= BC_FLOWCONTROLLER_MKS647C=Mks647c
			HEADERS *= $$PWD/mks647c.h
			SOURCES *= $$PWD/mks647c.cpp
			OPTHW *= "$${H}include <hardware/optional/flowcontroller/mks647c.h>"
		}
		equals(N,MKS946) {
		    DEFINES *= BC_FLOWCONTROLLER_MKS946=Mks946
			HEADERS *= $$PWD/mks946.h
			SOURCES *= $$PWD/mks946.cpp
			OPTHW *= "$${H}include <hardware/optional/flowcontroller/mks946.h>"
		}
	}
}
}
