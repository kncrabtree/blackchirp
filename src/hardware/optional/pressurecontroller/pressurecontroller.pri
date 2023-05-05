
HEADERS += $$PWD/pressurecontroller.h \
           $$PWD/pressurecontrollerconfig.h
SOURCES += $$PWD/pressurecontroller.cpp \
           $$PWD/pressurecontrollerconfig.cpp


count(PC,1) {

    N = $$upper($$PC)

    equals(N,VIRTUAL) {
	    DEFINES += BC_PCONTROLLER=VirtualPressureController BC_PCONTROLLER_H=virtualpressurecontroller.h
        HEADERS += $$PWD/virtualpressurecontroller.h
		SOURCES += $$PWD/virtualpressurecontroller.cpp
		}
	equals(N,INTELLISYS) {
	    DEFINES += BC_PCONTROLLER=IntellisysIQPlus BC_PCONTROLLER_H=intellisysiqplus.h
        HEADERS += $$PWD/intellisysiqplus.h
		SOURCES += $$PWD/intellisysiqplus.cpp
		}
}
