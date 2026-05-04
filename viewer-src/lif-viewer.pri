# LIF module components for Blackchirp Viewer
# Includes GUI, data, and storage components for LIF viewing
# Excludes hardware and acquisition components

# Always enable LIF for viewer since it's display-only
DEFINES += BC_LIF

SOURCES += $$PWD/../src/modules/lif/data/lifconfig.cpp \
    $$PWD/../src/modules/lif/data/liftrace.cpp \
    $$PWD/../src/modules/lif/data/lifstorage.cpp \
    $$PWD/../src/modules/lif/gui/lifdisplaywidget.cpp \
    $$PWD/../src/modules/lif/gui/lifsliceplot.cpp \
    $$PWD/../src/modules/lif/gui/liftraceplot.cpp \
    $$PWD/../src/modules/lif/gui/lifspectrogramplot.cpp \
    $$PWD/../src/modules/lif/gui/lifprocessingwidget.cpp \
    $$PWD/../src/modules/lif/hardware/lifdigitizer/lifdigitizerconfig.cpp

HEADERS += $$PWD/../src/modules/lif/data/lifconfig.h \
    $$PWD/../src/modules/lif/data/liftrace.h \
    $$PWD/../src/modules/lif/data/lifstorage.h \
    $$PWD/../src/modules/lif/gui/lifdisplaywidget.h \
    $$PWD/../src/modules/lif/gui/lifsliceplot.h \
    $$PWD/../src/modules/lif/gui/liftraceplot.h \
    $$PWD/../src/modules/lif/gui/lifspectrogramplot.h \
    $$PWD/../src/modules/lif/gui/lifprocessingwidget.h \
    $$PWD/../src/modules/lif/hardware/lifdigitizer/lifdigitizerconfig.h