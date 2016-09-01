SOURCES += mainwindow.cpp \
    $$PWD/ftmwviewwidget.cpp \
    $$PWD/ftplot.cpp \
    $$PWD/fidplot.cpp \
    $$PWD/communicationdialog.cpp \
    $$PWD/trackingviewwidget.cpp \
    $$PWD/trackingplot.cpp \
    $$PWD/zoompanplot.cpp \
    $$PWD/ftmwconfigwidget.cpp \
    $$PWD/chirpconfigwidget.cpp \
    $$PWD/chirptablemodel.cpp \
    $$PWD/chirpconfigplot.cpp \
    $$PWD/rfconfigwidget.cpp \
    $$PWD/pulseconfigwidget.cpp \
    $$PWD/pulseplot.cpp \
    $$PWD/led.cpp \
    $$PWD/ioboardconfigdialog.cpp \
    $$PWD/experimentviewwidget.cpp \
    $$PWD/ftmwsnapshotwidget.cpp \
    $$PWD/quickexptdialog.cpp \
    $$PWD/batchsequencedialog.cpp \
	$$PWD/exportbatchdialog.cpp

HEADERS += mainwindow.h \
    $$PWD/ftmwviewwidget.h \
    $$PWD/ftplot.h \
    $$PWD/fidplot.h \
    $$PWD/communicationdialog.h \
    $$PWD/trackingviewwidget.h \
    $$PWD/trackingplot.h \
    $$PWD/zoompanplot.h \
    $$PWD/ftmwconfigwidget.h \
    $$PWD/chirpconfigwidget.h \
    $$PWD/chirptablemodel.h \
    $$PWD/chirpconfigplot.h \
    $$PWD/rfconfigwidget.h \
    $$PWD/pulseconfigwidget.h \
    $$PWD/pulseplot.h \
    $$PWD/led.h \
    $$PWD/ioboardconfigdialog.h \
    $$PWD/experimentviewwidget.h \
    $$PWD/ftmwsnapshotwidget.h \
    $$PWD/quickexptdialog.h \
    $$PWD/batchsequencedialog.h \
	$$PWD/exportbatchdialog.h

FORMS += mainwindow.ui \
    $$PWD/ftmwviewwidget.ui \
    $$PWD/communicationdialog.ui \
    $$PWD/ftmwconfigwidget.ui \
    $$PWD/chirpconfigwidget.ui \
    $$PWD/rfconfigwidget.ui \
    $$PWD/pulseconfigwidget.ui \
    $$PWD/ioboardconfigdialog.ui \
    $$PWD/quickexptdialog.ui \
    $$PWD/batchsequencedialog.ui \
	$$PWD/exportbatchdialog.ui

lif {
	HEADERS += $$PWD/liftraceplot.h \
			   $$PWD/lifcontrolwidget.h \
			   $$PWD/lifdisplaywidget.h \
			   $$PWD/lifsliceplot.h \
			   $$PWD/lifspectrogramplot.h
	SOURCES += $$PWD/liftraceplot.cpp \
			   $$PWD/lifcontrolwidget.cpp \
			   $$PWD/lifdisplaywidget.cpp \
			   $$PWD/lifsliceplot.cpp \
			   $$PWD/lifspectrogramplot.cpp
	FORMS += $$PWD/lifcontrolwidget.ui \
			 $$PWD/lifdisplaywidget.ui
}

motor {
	HEADERS += $$PWD/motorscandialog.h \
			   $$PWD/motordisplaywidget.h \
			   $$PWD/motorsliderwidget.h \
			   $$PWD/motorspectrogramplot.h \
			   $$PWD/motorzplot.h \
			   $$PWD/motorxyplot.h \
			   $$PWD/motortimeplot.h
	SOURCES += $$PWD/motorscandialog.cpp \
			   $$PWD/motordisplaywidget.cpp \
			   $$PWD/motorsliderwidget.cpp \
			   $$PWD/motorspectrogramplot.cpp \
			   $$PWD/motorzplot.cpp \
			   $$PWD/motorxyplot.cpp \
			   $$PWD/motortimeplot.cpp
	FORMS += $$PWD/motorscandialog.ui \
			 $$PWD/motordisplaywidget.ui
}
