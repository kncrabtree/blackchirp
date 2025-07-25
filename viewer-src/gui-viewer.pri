# GUI components for Blackchirp Viewer
# Includes viewer main window and experiment viewing widgets
# Excludes acquisition UI, hardware config widgets, and experiment setup

SOURCES += $$PWD/viewermainwindow.cpp \
    $$PWD/../src/gui/overlay/unifiedoverlaywidget.cpp \
    $$PWD/../src/gui/overlay/unifiedoverlaydialog.cpp \
    $$PWD/../src/gui/overlay/overlaytypespecificwidget.cpp \
    $$PWD/../src/gui/overlay/bcexpoverlaywidget.cpp \
    $$PWD/../src/gui/overlay/catalogoverlaywidget.cpp \
    $$PWD/../src/gui/overlay/genericxyoverlaywidget.cpp \
    $$PWD/../src/gui/overlay/overlaybaseoptionswidget.cpp \
    $$PWD/../src/gui/overlay/overlaymanagerwidget.cpp \
    $$PWD/../src/gui/overlay/overlayconfiguredelegate.cpp \
    $$PWD/../src/gui/overlay/overlaycheckboxdelegate.cpp \
    $$PWD/../src/gui/dialog/peaklistexportdialog.cpp \
    $$PWD/../src/gui/plot/blackchirpplotcurve.cpp \
    $$PWD/../src/gui/plot/curveappearancewidget.cpp \
    $$PWD/../src/gui/plot/curveappearancepresetmanager.cpp \
    $$PWD/../src/gui/plot/presetsavedialog.cpp \
    $$PWD/../src/gui/plot/curvefactory.cpp \
    $$PWD/../src/gui/plot/customtracker.cpp \
    $$PWD/../src/gui/plot/customzoomer.cpp \
    $$PWD/../src/gui/plot/fidplot.cpp \
    $$PWD/../src/gui/plot/ftplot.cpp \
    $$PWD/../src/gui/plot/mainftplot.cpp \
    $$PWD/../src/gui/plot/trackingplot.cpp \
    $$PWD/../src/gui/plot/zoompanplot.cpp \
    $$PWD/../src/gui/widget/auxdataviewwidget.cpp \
    $$PWD/../src/gui/widget/experimentsummarywidget.cpp \
    $$PWD/../src/gui/widget/experimentviewwidget.cpp \
    $$PWD/../src/gui/widget/ftmwplottoolbar.cpp \
    $$PWD/../src/gui/widget/ftmwprocessingtoolbar.cpp \
    $$PWD/../src/gui/widget/ftmwviewwidget.cpp \
    $$PWD/../src/gui/widget/peakfindwidget.cpp \
    $$PWD/../src/gui/widget/scientificspinbox.cpp \
    $$PWD/../src/gui/widget/scientificinputwidget.cpp \
    $$PWD/../src/gui/widget/toolbarwidgetaction.cpp \
    $$PWD/../src/gui/style/themecolors.cpp

HEADERS += $$PWD/viewermainwindow.h \
    $$PWD/../src/gui/overlay/unifiedoverlaywidget.h \
    $$PWD/../src/gui/overlay/unifiedoverlaydialog.h \
    $$PWD/../src/gui/overlay/overlaytypespecificwidget.h \
    $$PWD/../src/gui/overlay/bcexpoverlaywidget.h \
    $$PWD/../src/gui/overlay/catalogoverlaywidget.h \
    $$PWD/../src/gui/overlay/genericxyoverlaywidget.h \
    $$PWD/../src/gui/overlay/overlaybaseoptionswidget.h \
    $$PWD/../src/gui/overlay/overlaymanagerwidget.h \
    $$PWD/../src/gui/overlay/overlayconfiguredelegate.h \
    $$PWD/../src/gui/overlay/overlaycheckboxdelegate.h \
    $$PWD/../src/gui/dialog/peaklistexportdialog.h \
    $$PWD/../src/gui/plot/blackchirpplotcurve.h \
    $$PWD/../src/gui/plot/curveappearancewidget.h \
    $$PWD/../src/gui/plot/curveappearancepresetmanager.h \
    $$PWD/../src/gui/plot/presetsavedialog.h \
    $$PWD/../src/gui/plot/curvefactory.h \
    $$PWD/../src/gui/plot/customtracker.h \
    $$PWD/../src/gui/plot/customzoomer.h \
    $$PWD/../src/gui/plot/fidplot.h \
    $$PWD/../src/gui/plot/ftplot.h \
    $$PWD/../src/gui/plot/mainftplot.h \
    $$PWD/../src/gui/plot/trackingplot.h \
    $$PWD/../src/gui/plot/zoompanplot.h \
    $$PWD/../src/gui/widget/auxdataviewwidget.h \
    $$PWD/../src/gui/widget/enumcombobox.h \
    $$PWD/../src/gui/widget/experimentsummarywidget.h \
    $$PWD/../src/gui/widget/experimentviewwidget.h \
    $$PWD/../src/gui/widget/ftmwplottoolbar.h \
    $$PWD/../src/gui/widget/ftmwprocessingtoolbar.h \
    $$PWD/../src/gui/widget/ftmwviewwidget.h \
    $$PWD/../src/gui/widget/peakfindwidget.h \
    $$PWD/../src/gui/widget/scientificspinbox.h \
    $$PWD/../src/gui/widget/scientificinputwidget.h \
    $$PWD/../src/gui/widget/toolbarwidgetaction.h \
    $$PWD/../src/gui/style/themecolors.h

FORMS += \
    $$PWD/../src/gui/widget/peakfindwidget.ui \
    $$PWD/../src/gui/dialog/peaklistexportdialog.ui