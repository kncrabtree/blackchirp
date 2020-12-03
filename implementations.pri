
CONFIG(allhardware) {
HEADERS += \
    $$PWD/virtualftmwscope.h \
    $$PWD/dsa71604c.h \
    $$PWD/virtualawg.h \
    $$PWD/virtualpulsegenerator.h \
    $$PWD/virtualflowcontroller.h \
    $$PWD/virtuallifscope.h \
    $$PWD/virtualgpibcontroller.h \
    $$PWD/awg70002a.h \
    $$PWD/qc9528.h \
    $$PWD/valon5009.h \
    $$PWD/mks647c.h \
    $$PWD/prologixgpiblan.h \
    $$PWD/virtualioboard.h \
    $$PWD/mso72004c.h \
    $$PWD/qc9518.h \
    $$PWD/awg7122b.h \
    $$PWD/virtualmotorcontroller.h \
    $$PWD/scx11.h \
    $$PWD/labjacku3.h \
    $$PWD/u3.h \
    $$PWD/virtualmotorscope.h \
    $$PWD/pico2206b.h \
    $$PWD/virtualpressurecontroller.h \
    $$PWD/intellisysiqplus.h \
    $$PWD/m4i2220x8.h \
    $$PWD/fixedclock.h \
    $$PWD/dsox92004a.h \
    $$PWD/m8195a.h \
    $$PWD/ad9914.h \
    $$PWD/valon5015.h \
    $$PWD/mks946.h \
    $$PWD/virtualtemperaturecontroller.h

SOURCES += \
    $$PWD/virtualftmwscope.cpp \
    $$PWD/dsa71604c.cpp \
    $$PWD/virtualawg.cpp \
    $$PWD/virtualpulsegenerator.cpp \
    $$PWD/virtualflowcontroller.cpp \
    $$PWD/virtuallifscope.cpp \
    $$PWD/virtualgpibcontroller.cpp \
    $$PWD/awg70002a.cpp \
    $$PWD/qc9528.cpp \
    $$PWD/valon5009.cpp \
    $$PWD/mks647c.cpp \
    $$PWD/prologixgpiblan.cpp \
    $$PWD/virtualioboard.cpp \
    $$PWD/mso72004c.cpp \
    $$PWD/qc9518.cpp \
    $$PWD/awg7122b.cpp \
    $$PWD/virtualmotorcontroller.cpp \
    $$PWD/scx11.cpp \
    $$PWD/labjacku3.cpp \
    $$PWD/u3.cpp \
    $$PWD/virtualmotorscope.cpp \
    $$PWD/pico2206b.cpp \
    $$PWD/virtualpressurecontroller.cpp  \
    $$PWD/intellisysiqplus.cpp \
    $$PWD/m4i2220x8.cpp \
    $$PWD/fixedclock.cpp \
    $$PWD/dsox92004a.cpp \
    $$PWD/m8195a.cpp \
    $$PWD/ad9914.cpp \
    $$PWD/valon5015.cpp \
    $$PWD/mks946.cpp \
    $$PWD/virtualtemperaturecontroller.cpp

} else {

equals(FTMWSCOPE,0) {
	HEADERS += $$PWD/virtualftmwscope.h
	SOURCES += $$PWD/virtualftmwscope.cpp
}

equals(FTMWSCOPE,1) {
	HEADERS += $$PWD/dsa71604c.h
	SOURCES += $$PWD/dsa71604c.cpp
}

equals(FTMWSCOPE,2) {
	HEADERS += $$PWD/mso72004c.h
	SOURCES += $$PWD/mso72004c.cpp
}

equals(FTMWSCOPE,3) {
     HEADERS += $$PWD/m4i2220x8.h
     SOURCES += $$PWD/m4i2220x8.cpp
}

equals(FTMWSCOPE,4) {
     HEADERS += $$PWD/dsox92004a.h
     SOURCES += $$PWD/dsox92004a.cpp
}


equals(AWG,0) {
	HEADERS += $$PWD/virtualawg.h
	SOURCES += $$PWD/virtualawg.cpp
}
equals(AWG,1) {
	HEADERS += $$PWD/awg70002a.h
	SOURCES += $$PWD/awg70002a.cpp
}
equals(AWG,2) {
	HEADERS += $$PWD/awg7122b.h
	SOURCES += $$PWD/awg7122b.cpp
}
equals(AWG,3) {
	HEADERS += $$PWD/ad9914.h
	SOURCES += $$PWD/ad9914.cpp
}
equals(AWG,4) {
	HEADERS += $$PWD/m8195a.h
	SOURCES += $$PWD/m8195a.cpp
}

contains(CLOCKS,0) {
        HEADERS += $$PWD/fixedclock.h
        SOURCES += $$PWD/fixedclock.cpp
}
contains(CLOCKS,1) {
        HEADERS += $$PWD/valon5009.h
        SOURCES += $$PWD/valon5009.cpp
}
contains(CLOCKS,2) {
        HEADERS += $$PWD/valon5015.h
        SOURCES += $$PWD/valon5015.cpp
}

equals(PGEN,0) {
	HEADERS += $$PWD/virtualpulsegenerator.h
	SOURCES += $$PWD/virtualpulsegenerator.cpp
}
equals(PGEN,1) {
	HEADERS += $$PWD/qc9528.h
	SOURCES += $$PWD/qc9528.cpp
}
equals(PGEN,2) {
	HEADERS += $$PWD/qc9518.h
	SOURCES += $$PWD/qc9518.cpp
}


equals(FC,0) {
	HEADERS += $$PWD/virtualflowcontroller.h
	SOURCES += $$PWD/virtualflowcontroller.cpp
}
equals(FC,1) {
	HEADERS += $$PWD/mks647c.h
	SOURCES += $$PWD/mks647c.cpp
}
equals(FC,2) {
    HEADERS += $$PWD/mks946.h
    SOURCES += $$PWD/mks946.cpp
}


equals(IOBOARD,0) {
	HEADERS += $$PWD/virtualioboard.h
	SOURCES += $$PWD/virtualioboard.cpp
}
equals(IOBOARD,1) {
	HEADERS += $$PWD/labjacku3.h \
			   $$PWD/u3.h
	SOURCES += $$PWD/labjacku3.cpp \
			   $$PWD/u3.cpp
	LIBS += -llabjackusb
}


equals(GPIB,0) {
	HEADERS += $$PWD/virtualgpibcontroller.h
	SOURCES += $$PWD/virtualgpibcontroller.cpp
}
equals(GPIB,1) {
	HEADERS += $$PWD/prologixgpiblan.h
	SOURCES += $$PWD/prologixgpiblan.cpp
}

equals(PC,0) {
        HEADERS += $$PWD/virtualpressurecontroller.h
        SOURCES += $$PWD/virtualpressurecontroller.cpp
}
equals(PC,1) {
        HEADERS += $$PWD/intellisysiqplus.h
        SOURCES += $$PWD/intellisysiqplus.cpp
}

equals(TC,0) {
        HEADERS += $$PWD/virtualtempcontroller.h
        SOURCES += $$PWD/virtualtempcontroller.cpp
}
equals(TC,1) {
#        HEADERS += $$PWD/intellisysiqplus.h
#        SOURCES += $$PWD/intellisysiqplus.cpp
}


lif {
	equals(LIFSCOPE,0) {
		HEADERS += $$PWD/virtuallifscope.h
		SOURCES += $$PWD/virtuallifscope.cpp
	}
}

motor {
	equals(MOTOR,0) {
                HEADERS += $$PWD/virtualmotorcontroller.h
                SOURCES += $$PWD/virtualmotorcontroller.cpp
	}
	equals(MOTOR,1) {
		HEADERS += $$PWD/scx11.h
		SOURCES += $$PWD/scx11.cpp
	}

        equals(MOTORSCOPE,0) {
                HEADERS += $$PWD/virtualmotorscope.h
                SOURCES += $$PWD/virtualmotorscope.cpp
        }
        equals(MOTORSCOPE,1) {
                HEADERS += $$PWD/pico2206b.h
                SOURCES += $$PWD/pico2206b.cpp
        }
}

}
