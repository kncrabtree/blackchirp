SOURCES += $$PWD/loghandler.cpp \
    $$PWD/datastructs.cpp \
    $$PWD/analysis/analysis.cpp \
    $$PWD/analysis/ft.cpp \
    $$PWD/analysis/ftworker.cpp \
    $$PWD/analysis/peakfinder.cpp \
    $$PWD/analysis/snapworker.cpp \
    $$PWD/experiment/chirpconfig.cpp \
    $$PWD/experiment/digitizerconfig.cpp \
    $$PWD/experiment/experiment.cpp \
    $$PWD/experiment/experimentobjective.cpp \
    $$PWD/experiment/fid.cpp \
    $$PWD/experiment/flowconfig.cpp \
    $$PWD/experiment/ftmwconfig.cpp \
    $$PWD/experiment/ftmwdigitizerconfig.cpp \
    $$PWD/experiment/ioboardconfig.cpp \
    $$PWD/experiment/pulsegenconfig.cpp \
    $$PWD/experiment/rfconfig.cpp \
    $$PWD/model/chirptablemodel.cpp \
    $$PWD/model/clocktablemodel.cpp \
    $$PWD/model/ioboardconfigmodel.cpp \
    $$PWD/model/peaklistmodel.cpp \
	$$PWD/model/validationmodel.cpp \
    $$PWD/storage/blackchirpcsv.cpp \
    $$PWD/storage/fidpeakupstorage.cpp \
    $$PWD/storage/fidsinglestorage.cpp \
    $$PWD/storage/fidstoragebase.cpp \
    $$PWD/storage/headerstorage.cpp \
   $$PWD/storage/settingsstorage.cpp


HEADERS += $$PWD/loghandler.h \
    $$PWD/datastructs.h \
    $$PWD/analysis/analysis.h \
    $$PWD/analysis/ft.h \
    $$PWD/analysis/ftworker.h \
    $$PWD/analysis/peakfinder.h \
    $$PWD/analysis/snapworker.h \
    $$PWD/experiment/chirpconfig.h \
    $$PWD/experiment/digitizerconfig.h \
    $$PWD/experiment/experiment.h \
    $$PWD/experiment/experimentobjective.h \
    $$PWD/experiment/fid.h \
    $$PWD/experiment/flowconfig.h \
    $$PWD/experiment/ftmwconfig.h \
    $$PWD/experiment/ftmwdigitizerconfig.h \
    $$PWD/experiment/ioboardconfig.h \
    $$PWD/experiment/pulsegenconfig.h \
    $$PWD/experiment/rfconfig.h \
    $$PWD/model/chirptablemodel.h \
    $$PWD/model/clocktablemodel.h \
    $$PWD/model/ioboardconfigmodel.h \
    $$PWD/model/peaklistmodel.h \
	$$PWD/model/validationmodel.h \
    $$PWD/storage/blackchirpcsv.h \
    $$PWD/storage/fidpeakupstorage.h \
    $$PWD/storage/fidsinglestorage.h \
    $$PWD/storage/fidstoragebase.h \
    $$PWD/storage/headerstorage.h \
   $$PWD/storage/settingsstorage.h

DISTFILES += \
   $$PWD/../../doc/Doxyfile \
   $$PWD/../../doc/Makefile \
   $$PWD/../../doc/make.bat
