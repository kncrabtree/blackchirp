
HEADERS += $$PWD/flowcontroller.h \
           $$PWD/flowconfig.h
SOURCES += $$PWD/flowcontroller.cpp \
           $$PWD/flowconfig.cpp


!lessThan(FC,0) {
    DEFINES += BC_FLOWCONTROLLER=$$FC

    equals(FC,0) {
	    HEADERS += $$PWD/virtualflowcontroller.h
		SOURCES += $$PWD/virtualflowcontroller.h
	}
	equals(FC,1) {
	    HEADERS += $$PWD/mks647c.h
		SOURCES += $$PWD/mks647c.cpp
		}
	equals(FC,2) {
	    HEADERS += $$PWD/mks946.h
		SOURCES += $$PWD/mks946.cpp
	}
}
