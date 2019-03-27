SOURCES += mainwindow.cpp \
    $$PWD/ftmwviewwidget.cpp \
    $$PWD/ftplot.cpp \
    $$PWD/fidplot.cpp \
    $$PWD/communicationdialog.cpp \
    $$PWD/trackingviewwidget.cpp \
    $$PWD/trackingplot.cpp \
    $$PWD/zoompanplot.cpp \
    $$PWD/chirpconfigwidget.cpp \
    $$PWD/chirptablemodel.cpp \
    $$PWD/chirpconfigplot.cpp \
    $$PWD/pulseconfigwidget.cpp \
    $$PWD/pulseplot.cpp \
    $$PWD/led.cpp \
    $$PWD/ioboardconfigdialog.cpp \
    $$PWD/experimentviewwidget.cpp \
    $$PWD/quickexptdialog.cpp \
    $$PWD/batchsequencedialog.cpp \
    $$PWD/exportbatchdialog.cpp \
    $$PWD/peakfindwidget.cpp \
    $$PWD/peaklistmodel.cpp \
    $$PWD/peaklistexportdialog.cpp \
    $$PWD/rfconfigwidget.cpp \
    $$PWD/clocktablemodel.cpp \
    $$PWD/clockdisplaywidget.cpp \
    $$PWD/digitizerconfigwidget.cpp \
    $$PWD/ftmwprocessingwidget.cpp \
    $$PWD/ftmwplotconfigwidget.cpp

HEADERS += mainwindow.h \
    $$PWD/ftmwviewwidget.h \
    $$PWD/ftplot.h \
    $$PWD/fidplot.h \
    $$PWD/communicationdialog.h \
    $$PWD/trackingviewwidget.h \
    $$PWD/trackingplot.h \
    $$PWD/zoompanplot.h \
    $$PWD/chirpconfigwidget.h \
    $$PWD/chirptablemodel.h \
    $$PWD/chirpconfigplot.h \
    $$PWD/pulseconfigwidget.h \
    $$PWD/pulseplot.h \
    $$PWD/led.h \
    $$PWD/ioboardconfigdialog.h \
    $$PWD/experimentviewwidget.h \
    $$PWD/quickexptdialog.h \
    $$PWD/batchsequencedialog.h \
    $$PWD/exportbatchdialog.h \
    $$PWD/peakfindwidget.h \
    $$PWD/peaklistmodel.h \
    $$PWD/peaklistexportdialog.h \
    $$PWD/rfconfigwidget.h \
    $$PWD/clocktablemodel.h \
    $$PWD/clockdisplaywidget.h \
    $$PWD/digitizerconfigwidget.h \
    $$PWD/ftmwprocessingwidget.h \
    $$PWD/ftmwplotconfigwidget.h

FORMS += mainwindow.ui \
    $$PWD/communicationdialog.ui \
    $$PWD/chirpconfigwidget.ui \
    $$PWD/pulseconfigwidget.ui \
    $$PWD/ioboardconfigdialog.ui \
    $$PWD/quickexptdialog.ui \
    $$PWD/batchsequencedialog.ui \
    $$PWD/exportbatchdialog.ui \
    $$PWD/peakfindwidget.ui \
    $$PWD/peaklistexportdialog.ui \
    $$PWD/rfconfigwidget.ui \
    $$PWD/digitizerconfigwidget.ui

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
	FORMS +=   $$PWD/lifcontrolwidget.ui \
			   $$PWD/lifdisplaywidget.ui
}

motor {
	HEADERS += \
			   $$PWD/motordisplaywidget.h \
			   $$PWD/motorsliderwidget.h \
			   $$PWD/motorspectrogramplot.h \
			   $$PWD/motorzplot.h \
			   $$PWD/motorxyplot.h \
			   $$PWD/motortimeplot.h \
			   $$PWD/motorscopeconfigwidget.h \
			   $$PWD/motorscanconfigwidget.h \
			   $$PWD/motorstatuswidget.h
	SOURCES += \
			   $$PWD/motordisplaywidget.cpp \
			   $$PWD/motorsliderwidget.cpp \
			   $$PWD/motorspectrogramplot.cpp \
			   $$PWD/motorzplot.cpp \
			   $$PWD/motorxyplot.cpp \
			   $$PWD/motortimeplot.cpp \
			   $$PWD/motorscopeconfigwidget.cpp \
			   $$PWD/motorscanconfigwidget.cpp \
			   $$PWD/motorstatuswidget.cpp

	FORMS += \
			   $$PWD/motorscopeconfigwidget.ui \
			   $$PWD/motorscanconfigwidget.ui \
			   $$PWD/motorstatuswidget.ui
}
