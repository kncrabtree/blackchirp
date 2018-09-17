HEADERS += \
    $$PWD/experimentwizard.h \
    $$PWD/wizardstartpage.h \
    $$PWD/wizardchirpconfigpage.h \
    $$PWD/wizardsummarypage.h \
    $$PWD/wizardpulseconfigpage.h \
    $$PWD/wizardvalidationpage.h \
    $$PWD/ioboardconfigmodel.h \
    $$PWD/validationmodel.h \
    $$PWD/wizardrfconfigpage.h \
    $$PWD/experimentwizardpage.h \
    $$PWD/wizarddigitizerconfigpage.h

SOURCES += \
    $$PWD/experimentwizard.cpp \
    $$PWD/wizardstartpage.cpp \
    $$PWD/wizardchirpconfigpage.cpp \
    $$PWD/wizardsummarypage.cpp \
    $$PWD/wizardpulseconfigpage.cpp \
    $$PWD/wizardvalidationpage.cpp \
    $$PWD/ioboardconfigmodel.cpp \
    $$PWD/validationmodel.cpp \
    $$PWD/wizardrfconfigpage.cpp \
    $$PWD/experimentwizardpage.cpp \
    $$PWD/wizarddigitizerconfigpage.cpp

lif {
	HEADERS += $$PWD/wizardlifconfigpage.h
	SOURCES += $$PWD/wizardlifconfigpage.cpp
}

motor {
        HEADERS += $$PWD/wizardmotorscanconfigpage.h
        SOURCES += $$PWD/wizardmotorscanconfigpage.cpp
}
