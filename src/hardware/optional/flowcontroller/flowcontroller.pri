
HEADERS += $$PWD/flowcontroller.h \
           $$PWD/flowconfig.h
SOURCES += $$PWD/flowcontroller.cpp \
           $$PWD/flowconfig.cpp



count(FC,1) {
    N = $$upper($$FC)

    equals(N,VIRTUAL) {
	    DEFINES += BC_FLOWCONTROLLER=VirtualFlowController BC_FLOWCONTROLLER_H=virtualflowcontroller.h
	    HEADERS += $$PWD/virtualflowcontroller.h
		SOURCES += $$PWD/virtualflowcontroller.cpp
	}
	equals(N,MKS647C) {
	    DEFINES += BC_FLOWCONTROLLER=Mks647c BC_FLOWCONTROLLER_H=mks647c.h
	    HEADERS += $$PWD/mks647c.h
		SOURCES += $$PWD/mks647c.cpp
		}
	equals(N,MKS946) {
	    DEFINES += BC_FLOWCONTROLLER=Mks946 BC_FLOWCONTROLLER_H=mks946.h
	    HEADERS += $$PWD/mks946.h
		SOURCES += $$PWD/mks946.cpp
	}
}
