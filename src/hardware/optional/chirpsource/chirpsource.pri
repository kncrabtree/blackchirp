HEADERS += $$PWD/awg.h
SOURCES += $$PWD/awg.cpp

count(AWG,1) {

    N = $$upper($$AWG)

    equals(N,VIRTUAL) {
            DEFINES += BC_AWG=VirtualAwg
	    HEADERS += $$PWD/virtualawg.h
            SOURCES += $$PWD/virtualawg.cpp
            HW *= "$${H}include <hardware/optional/chirpsource/virtualawg.h>"
	}
	equals(N,AWG70002A) {
            DEFINES += BC_AWG=AWG70002a
	    HEADERS += $$PWD/awg70002a.h
            SOURCES += $$PWD/awg70002a.cpp
            HW *= "$${H}include <hardware/optional/chirpsource/awg70002a.h>"
	}
	equals(N,AWG7122B) {
            DEFINES += BC_AWG=AWG7122B
	    HEADERS += $$PWD/awg7122b.h
            SOURCES += $$PWD/awg7122b.cpp
            HW *= "$${H}include <hardware/optional/chirpsource/awg7122b.h>"
	}
	equals(N,AD9914) {
            DEFINES += BC_AWG=AD9914
	    HEADERS += $$PWD/ad9914.h
            SOURCES += $$PWD/ad9914.cpp
            HW *= "$${H}include <hardware/optional/chirpsource/ad9914.h>"
        }
	equals(N,M8195A) {
            DEFINES += BC_AWG=M8195A
	    HEADERS += $$PWD/m8195a.h
            SOURCES += $$PWD/m8195a.cpp
            HW *= "$${H}include <hardware/optional/chirpsource/m8195a.h>"
	}
        equals(N,AWG5204) {
            DEFINES += BC_AWG=AWG5204
            HEADERS += $$PWD/awg5204.h
            SOURCES += $$PWD/awg5204.cpp
            HW *= "$${H}include <hardware/optional/chirpsource/awg5204.h>"
        }
		equals(N,M8190) {
		    DEFINES += BC_AWG=M8190
            HEADERS += $$PWD/m8190.h
			SOURCES += $$PWD/m8190.cpp
			HW *= "$${H}include <hardware/optional/chirpsource/m8190.h>"
        }
}

allhardware {
    HEADERS *= $$PWD/virtualawg.h
    SOURCES *= $$PWD/virtualawg.cpp
    HW *= "$${H}include <hardware/optional/chirpsource/virtualawg.h>"
    HEADERS *= $$PWD/awg70002a.h
    SOURCES *= $$PWD/awg70002a.cpp
    HW *= "$${H}include <hardware/optional/chirpsource/awg70002a.h>"
    HEADERS *= $$PWD/awg7122b.h
    SOURCES *= $$PWD/awg7122b.cpp
    HW *= "$${H}include <hardware/optional/chirpsource/awg7122b.h>"
    HEADERS *= $$PWD/ad9914.h
    SOURCES *= $$PWD/ad9914.cpp
    HW *= "$${H}include <hardware/optional/chirpsource/ad9914.h>"
    HEADERS *= $$PWD/m8195a.h
    SOURCES *= $$PWD/m8195a.cpp
    HW *= "$${H}include <hardware/optional/chirpsource/m8195a.h>"
    HEADERS *= $$PWD/awg5204.h
    SOURCES *= $$PWD/awg5204.cpp
    HW *= "$${H}include <hardware/optional/chirpsource/awg5204.h>"
	HEADERS += $$PWD/m8190.h
	SOURCES += $$PWD/m8190.cpp
	HW *= "$${H}include <hardware/optional/chirpsource/m8190.h>"
}
