SOURCES += $$PWD/loghandler.cpp \
    $$PWD/analysis/analysis.cpp \
    $$PWD/analysis/ft.cpp \
    $$PWD/analysis/ftworker.cpp \
    $$PWD/analysis/peakfinder.cpp \
    $$PWD/experiment/chirpconfig.cpp \
    $$PWD/experiment/digitizerconfig.cpp \
    $$PWD/experiment/experiment.cpp \
    $$PWD/experiment/experimentobjective.cpp \
    $$PWD/experiment/experimentvalidator.cpp \
    $$PWD/experiment/fid.cpp \
    $$PWD/experiment/ftmwconfig.cpp \
    $$PWD/experiment/ftmwconfigtypes.cpp \
    $$PWD/experiment/rfconfig.cpp \
    $$PWD/model/chirptablemodel.cpp \
    $$PWD/model/clocktablemodel.cpp \
   $$PWD/model/exptsummarymodel.cpp \
   $$PWD/model/hwsettingsmodel.cpp \
    $$PWD/model/peaklistmodel.cpp \
	$$PWD/model/validationmodel.cpp \
    $$PWD/storage/auxdatastorage.cpp \
    $$PWD/storage/blackchirpcsv.cpp \
    $$PWD/storage/datastoragebase.cpp \
   $$PWD/storage/fidmultistorage.cpp \
    $$PWD/storage/fidpeakupstorage.cpp \
    $$PWD/storage/fidsinglestorage.cpp \
    $$PWD/storage/fidstoragebase.cpp \
    $$PWD/storage/headerstorage.cpp \
   $$PWD/storage/settingsstorage.cpp


HEADERS += $$PWD/loghandler.h \
    $$PWD/analysis/analysis.h \
    $$PWD/analysis/ft.h \
    $$PWD/analysis/ftworker.h \
    $$PWD/analysis/peakfinder.h \
    $$PWD/experiment/chirpconfig.h \
    $$PWD/experiment/digitizerconfig.h \
    $$PWD/experiment/experiment.h \
    $$PWD/experiment/experimentobjective.h \
    $$PWD/experiment/experimentvalidator.h \
    $$PWD/experiment/fid.h \
    $$PWD/experiment/ftmwconfig.h \
    $$PWD/experiment/ftmwconfigtypes.h \
    $$PWD/experiment/rfconfig.h \
    $$PWD/model/chirptablemodel.h \
    $$PWD/model/clocktablemodel.h \
   $$PWD/model/exptsummarymodel.h \
   $$PWD/model/hwsettingsmodel.h \
    $$PWD/model/peaklistmodel.h \
	$$PWD/model/validationmodel.h \
    $$PWD/storage/auxdatastorage.h \
    $$PWD/storage/blackchirpcsv.h \
    $$PWD/storage/datastoragebase.h \
   $$PWD/storage/fidmultistorage.h \
    $$PWD/storage/fidpeakupstorage.h \
    $$PWD/storage/fidsinglestorage.h \
    $$PWD/storage/fidstoragebase.h \
    $$PWD/storage/headerstorage.h \
   $$PWD/storage/settingsstorage.h

DISTFILES += \
   $$PWD/../../doc/Doxyfile \
   $$PWD/../../doc/Makefile \
   $$PWD/../../doc/make.bat
