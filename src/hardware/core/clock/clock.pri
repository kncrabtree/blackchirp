HEADERS += \
    $$PWD/clock.h \
    $$PWD/clockmanager.h

SOURCES += \
    $$PWD/clock.cpp \
    $$PWD/clockmanager.cpp


DEFINES += BC_NUM_CLOCKS=$$size(CLOCKS)
for(num, 0..$$size(CLOCKS)) {
   N = $$upper($$member(CLOCKS,$$num))
   count(N,1) {
      DEFINES += BC_CLOCK_$$num=BC_CLOCK_$$N
	  equals(N,FIXED) {
	     DEFINES *= BC_CLOCK_FIXED=FixedClock
		 HEADERS *= $$PWD/fixedclock.h
		 SOURCES *= $$PWD/fixedclock.cpp
		}
	  equals(N,VALON5009) {
	     DEFINES *= BC_CLOCK_VALON5009=Valon5009
		 HEADERS *= $$PWD/valon5009.h
		 SOURCES *= $$PWD/valon5009.cpp
		}
	  equals(N,VALON5015) {
	     DEFINES *= BC_CLOCK_VALON5015=Valon5015
		 HEADERS *= $$PWD/valon5015.h
		 SOURCES *= $$PWD/valon5015.cpp
		}
	  equals(N,HP83712B) {
	     DEFINES *= BC_CLOCK_HP83712B=HP83712B
		 HEADERS *= $$PWD/hp83712b.h
		 SOURCES *= $$PWD/hp83712b.cpp
		}
   }
}


