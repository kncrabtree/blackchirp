!lessThan(AWG,0) {
    HEADERS += $$PWD/awg.h
    SOURCES += $$PWD/awg.cpp

    DEFINES += BC_AWG=$$AWG
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
        equals(AWG,5) {
            HEADERS += $$PWD/awg5204.h
                SOURCES += $$PWD/awg5204.cpp
        }
        equals(AWG,6) {
            HEADERS += $$PWD/m8190.h
                SOURCES += $$PWD/m8190.cpp
        }
}
