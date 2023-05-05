HEADERS += $$PWD/awg.h
SOURCES += $$PWD/awg.cpp

count(AWG,1) {

    N = $$upper($$AWG)

    equals(N,VIRTUAL) {
	    DEFINES += BC_AWG=VirtualAwg BC_AWG_H=virtualawg.h
	    HEADERS += $$PWD/virtualawg.h
		SOURCES += $$PWD/virtualawg.cpp
	}
	equals(N,AWG70002A) {
	    DEFINES += BC_AWG=AWG70002a BC_AWG_H=awg70002a.h
	    HEADERS += $$PWD/awg70002a.h
		SOURCES += $$PWD/awg70002a.cpp
	}
	equals(N,AWG7122B) {
	    DEFINES += BC_AWG=AWG7122B BC_AWG_H=awg7122b.h
	    HEADERS += $$PWD/awg7122b.h
		SOURCES += $$PWD/awg7122b.cpp
	}
	equals(N,AD9914) {
	    DEFINES += BC_AWG=AD9914 BC_AWG_H=ad9914.h
	    HEADERS += $$PWD/ad9914.h
		SOURCES += $$PWD/ad9914.cpp
		}
	equals(N,M8195A) {
	    DEFINES += BC_AWG=M8195A BC_AWG_H=m8195a.h
	    HEADERS += $$PWD/m8195a.h
		SOURCES += $$PWD/m8195a.cpp
	}
	    equals(N,AWG5204) {
		    DEFINES += BC_AWG=AWG5204 BC_AWG_H=awg5204.h
            HEADERS += $$PWD/awg5204.h
            SOURCES += $$PWD/awg5204.cpp
        }
}
