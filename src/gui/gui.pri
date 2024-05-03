SOURCES += $$PWD/mainwindow.cpp \
    $$PWD/dialog/batchsequencedialog.cpp \
    $$PWD/dialog/bcsavepathdialog.cpp \
    $$PWD/dialog/communicationdialog.cpp \
   $$PWD/dialog/hwdialog.cpp \
    $$PWD/dialog/peaklistexportdialog.cpp \
    $$PWD/dialog/quickexptdialog.cpp \
    $$PWD/expsetup/experimentchirpconfigpage.cpp \
    $$PWD/expsetup/experimentconfigpage.cpp \
    $$PWD/expsetup/experimentdrscanconfigpage.cpp \
    $$PWD/expsetup/experimentflowconfigpage.cpp \
    $$PWD/expsetup/experimentftmwdigitizerconfigpage.cpp \
    $$PWD/expsetup/experimentioboardconfigpage.cpp \
    $$PWD/expsetup/experimentloscanconfigpage.cpp \
    $$PWD/expsetup/experimentpressurecontrollerconfigpage.cpp \
    $$PWD/expsetup/experimentpulsegenconfigpage.cpp \
    $$PWD/expsetup/experimentrfconfigpage.cpp \
    $$PWD/expsetup/experimentsetupdialog.cpp \
    $$PWD/expsetup/experimenttemperaturecontrollerconfigpage.cpp \
    $$PWD/expsetup/experimenttypepage.cpp \
    $$PWD/expsetup/experimentvalidatorconfigpage.cpp \
    $$PWD/plot/blackchirpplotcurve.cpp \
    $$PWD/plot/chirpconfigplot.cpp \
    $$PWD/plot/customtracker.cpp \
    $$PWD/plot/customzoomer.cpp \
    $$PWD/plot/fidplot.cpp \
    $$PWD/plot/ftplot.cpp \
    $$PWD/plot/pulseplot.cpp \
    $$PWD/plot/trackingplot.cpp \
    $$PWD/plot/zoompanplot.cpp \
    $$PWD/widget/auxdataviewwidget.cpp \
    $$PWD/widget/chirpconfigwidget.cpp \
    $$PWD/widget/clockdisplaybox.cpp \
    $$PWD/widget/digitizerconfigwidget.cpp \
   $$PWD/widget/experimentsummarywidget.cpp \
    $$PWD/widget/experimentviewwidget.cpp \
    $$PWD/widget/ftmwdigitizerconfigwidget.cpp \
    $$PWD/widget/ftmwplottoolbar.cpp \
    $$PWD/widget/ftmwprocessingtoolbar.cpp \
    $$PWD/widget/ftmwviewwidget.cpp \
    $$PWD/widget/gascontrolwidget.cpp \
    $$PWD/widget/gasflowdisplaywidget.cpp \
    $$PWD/widget/hardwarestatusbox.cpp \
    $$PWD/widget/ioboardconfigwidget.cpp \
    $$PWD/widget/led.cpp \
    $$PWD/widget/peakfindwidget.cpp \
   $$PWD/widget/pressurecontrolwidget.cpp \
    $$PWD/widget/pressurestatusbox.cpp \
    $$PWD/widget/pulseconfigwidget.cpp \
   $$PWD/widget/pulsestatusbox.cpp \
    $$PWD/widget/rfconfigwidget.cpp \
    $$PWD/widget/temperaturecontrolwidget.cpp \
    $$PWD/widget/temperaturestatusbox.cpp \
    $$PWD/widget/toolbarwidgetaction.cpp

HEADERS += $$PWD/mainwindow.h \
    $$PWD/dialog/batchsequencedialog.h \
    $$PWD/dialog/bcsavepathdialog.h \
    $$PWD/dialog/communicationdialog.h \
   $$PWD/dialog/hwdialog.h \
    $$PWD/dialog/peaklistexportdialog.h \
    $$PWD/dialog/quickexptdialog.h \
    $$PWD/expsetup/experimentchirpconfigpage.h \
    $$PWD/expsetup/experimentconfigpage.h \
    $$PWD/expsetup/experimentdrscanconfigpage.h \
    $$PWD/expsetup/experimentflowconfigpage.h \
    $$PWD/expsetup/experimentftmwdigitizerconfigpage.h \
    $$PWD/expsetup/experimentioboardconfigpage.h \
    $$PWD/expsetup/experimentloscanconfigpage.h \
    $$PWD/expsetup/experimentpressurecontrollerconfigpage.h \
    $$PWD/expsetup/experimentpulsegenconfigpage.h \
    $$PWD/expsetup/experimentrfconfigpage.h \
    $$PWD/expsetup/experimentsetupdialog.h \
    $$PWD/expsetup/experimenttemperaturecontrollerconfigpage.h \
    $$PWD/expsetup/experimenttypepage.h \
    $$PWD/expsetup/experimentvalidatorconfigpage.h \
    $$PWD/mainwindow_ui.h \
    $$PWD/plot/blackchirpplotcurve.h \
    $$PWD/plot/chirpconfigplot.h \
    $$PWD/plot/customtracker.h \
    $$PWD/plot/customzoomer.h \
    $$PWD/plot/fidplot.h \
    $$PWD/plot/ftplot.h \
    $$PWD/plot/pulseplot.h \
    $$PWD/plot/trackingplot.h \
    $$PWD/plot/zoompanplot.h \
    $$PWD/widget/auxdataviewwidget.h \
    $$PWD/widget/chirpconfigwidget.h \
    $$PWD/widget/clockdisplaybox.h \
    $$PWD/widget/digitizerconfigwidget.h \
    $$PWD/widget/enumcombobox.h \
   $$PWD/widget/experimentsummarywidget.h \
    $$PWD/widget/experimentviewwidget.h \
    $$PWD/widget/ftmwdigitizerconfigwidget.h \
    $$PWD/widget/ftmwplottoolbar.h \
    $$PWD/widget/ftmwprocessingtoolbar.h \
    $$PWD/widget/ftmwviewwidget.h \
    $$PWD/widget/gascontrolwidget.h \
    $$PWD/widget/gasflowdisplaywidget.h \
    $$PWD/widget/hardwarestatusbox.h \
    $$PWD/widget/ioboardconfigwidget.h \
    $$PWD/widget/led.h \
    $$PWD/widget/peakfindwidget.h \
   $$PWD/widget/pressurecontrolwidget.h \
    $$PWD/widget/pressurestatusbox.h \
    $$PWD/widget/pulseconfigwidget.h \
   $$PWD/widget/pulsestatusbox.h \
    $$PWD/widget/rfconfigwidget.h \
    $$PWD/widget/temperaturecontrolwidget.h \
    $$PWD/widget/temperaturestatusbox.h \
    $$PWD/widget/toolbarwidgetaction.h

FORMS += \
    $$PWD/dialog/communicationdialog.ui \
    $$PWD/dialog/peaklistexportdialog.ui \
    $$PWD/widget/chirpconfigwidget.ui \
    $$PWD/widget/digitizerconfigwidget.ui \
    $$PWD/widget/peakfindwidget.ui \
    $$PWD/widget/rfconfigwidget.ui
