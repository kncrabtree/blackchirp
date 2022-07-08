#ifndef MAINWINDOW_UI_H
#define MAINWINDOW_UI_H


#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <QtWidgets/QSpacerItem>
#include <gui/widget/ftmwviewwidget.h>
#include <gui/widget/led.h>
#include <gui/widget/auxdataviewwidget.h>
#include <gui/widget/clockdisplaybox.h>
#include <gui/widget/toolbarwidgetaction.h>

#ifdef BC_LIF
#include <modules/lif/gui/lifdisplaywidget.h>
#endif

class Ui_MainWindow
{
public:
    QAction *actionStart_Experiment;
    QToolButton *acquireButton;
    QToolButton *pauseButton;
    QToolButton *resumeButton;
    QToolButton *abortButton;
    QToolButton *sleepButton;
    QToolButton *auxPlotButton;
    QToolButton *rollingPlotButton;
    QAction *actionCommunication;
    SpinBoxWidgetAction *auxGraphsBox;
    SpinBoxWidgetAction *rollingGraphsBox;
    QAction *actionTest_All_Connections;
    QAction *actionAutoscale_Rolling;
    QAction *actionAutoscale_Aux;
    SpinBoxWidgetAction *rollingDurationBox;
    QAction *actionView_Experiment;
    QAction *actionQuick_Experiment;
    QAction *actionStart_Sequence;
    QAction *actionRfConfig;
    QToolButton *hardwareButton;
    QToolButton *settingsButton;
    QAction *fontAction;
    QAction *savePathAction;
    QWidget *centralWidget;
    QHBoxLayout *mainLayout;
    QVBoxLayout *instrumentStatusLayout;
    QLabel *instStatusLabel;
    QGridLayout *statusLayout;
    QLabel *exptLabel;
    QSpinBox *exptSpinBox;
    ClockDisplayBox *clockBox;
    QSpacerItem *statusSpacer;
    QLabel *ftmwProgressLabel;
    QProgressBar *ftmwProgressBar;
    QTabWidget *mainTabWidget;
    QWidget *ftmwTab;
    QVBoxLayout *ftmwTabLayout;
    FtmwViewWidget *ftViewWidget;
    QWidget *auxDataTab;
    QVBoxLayout *auxDataViewLayout;
    AuxDataViewWidget *auxDataViewWidget;
    QWidget *rollingDataTab;
    QVBoxLayout *rollingDataViewLayout;
    RollingDataWidget *rollingDataViewWidget;
    QWidget *logTab;
    QVBoxLayout *logTabLayout;
    QTextEdit *logTextEdit;
    QMenu *menuHardware;
    QMenu *menuAcquisition;
    QMenu *menuRollingData;
    QMenu *menuAuxData;
    QMenu *settingsMenu;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;


#ifdef BC_LIF
    QWidget *lifTab;
    QAction *actionLifConfig;
    QProgressBar *lifProgressBar;
    QLayout *lifTabLayout;
    LifDisplayWidget *lifDisplayWidget;
#endif

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/bc_logo_small.png"), QSize(), QIcon::Normal, QIcon::Off);
        MainWindow->setWindowIcon(icon);
        actionStart_Experiment = new QAction(MainWindow);
        actionStart_Experiment->setObjectName(QString::fromUtf8("actionStart_Experiment"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/experiment.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionStart_Experiment->setIcon(icon1);
        pauseButton = new QToolButton(MainWindow);
        pauseButton->setObjectName(QString::fromUtf8("actionPause"));
        pauseButton->setToolTip("Pause acquisition");
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/icons/pause.png"), QSize(), QIcon::Normal, QIcon::Off);
        pauseButton->setIcon(icon2);
        resumeButton = new QToolButton(MainWindow);
        resumeButton->setObjectName(QString::fromUtf8("actionResume"));
        resumeButton->setToolTip("Resume acquisition");
        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/icons/start.png"), QSize(), QIcon::Normal, QIcon::Off);
        resumeButton->setIcon(icon3);
        abortButton = new QToolButton(MainWindow);
        abortButton->setObjectName(QString::fromUtf8("actionAbort"));
        abortButton->setToolTip("Abort acquistiion");
        QIcon icon4;
        icon4.addFile(QString::fromUtf8(":/icons/abort.png"), QSize(), QIcon::Normal, QIcon::Off);
        abortButton->setIcon(icon4);
        actionCommunication = new QAction(MainWindow);
        actionCommunication->setObjectName(QString::fromUtf8("actionCommunication"));
        QIcon icon5;
        icon5.addFile(QString::fromUtf8(":/icons/computer.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionCommunication->setIcon(icon5);
        QIcon auxIcon;
        auxIcon.addFile(QString::fromUtf8(":/icons/dataplots.png"), QSize(), QIcon::Normal, QIcon::Off);
        QIcon rollIcon;
        rollIcon.addFile(QString(":/icons/view-media-visualization.svg"), QSize(), QIcon::Normal, QIcon::Off);
        QIcon icon7;
        icon7.addFile(QString::fromUtf8(":/icons/num.png"), QSize(), QIcon::Normal, QIcon::Off);
        QIcon icon8;
        icon8.addFile(QString::fromUtf8(":/icons/connect.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionTest_All_Connections = new QAction(MainWindow);
        actionTest_All_Connections->setObjectName(QString::fromUtf8("actionTest_All_Connections"));
        actionTest_All_Connections->setIcon(icon8);
        sleepButton = new QToolButton(MainWindow);
        sleepButton->setObjectName(QString::fromUtf8("actionSleep"));
        sleepButton->setCheckable(true);
        QIcon icon9;
        icon9.addFile(QString::fromUtf8(":/icons/sleep.png"), QSize(), QIcon::Normal, QIcon::Off);
        sleepButton->setIcon(icon9);
        sleepButton->setToolTip("Toggle sleep mode, putting hardware in a standby state.\nIf pressed during an acquisition, sleep mode will be activated when the acquisition is complete.");
        actionAutoscale_Aux = new QAction(MainWindow);
        actionAutoscale_Aux->setObjectName(QString::fromUtf8("actionAutoscale_Aux"));
        QIcon icon10;
        icon10.addFile(QString::fromUtf8(":/icons/autoscale.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionAutoscale_Aux->setIcon(icon10);
        actionAutoscale_Rolling = new QAction(MainWindow);
        actionAutoscale_Rolling->setObjectName(QString::fromUtf8("actionAutoscale_Rolling"));
        actionAutoscale_Rolling->setIcon(icon10);
        QIcon rfIcon;
        rfIcon.addFile(QString::fromUtf8(":/icons/chirp.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionRfConfig = new QAction("Rf Configuration",MainWindow);
        actionRfConfig->setObjectName("ActionRfConfig");
        actionRfConfig->setIcon(rfIcon);
        QIcon icon12;
        icon12.addFile(QString::fromUtf8(":/icons/controltab.png"), QSize(), QIcon::Normal, QIcon::Off);
        QIcon icon13;
        icon13.addFile(QString::fromUtf8(":/icons/log.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionView_Experiment = new QAction(MainWindow);
        actionView_Experiment->setObjectName(QString::fromUtf8("actionView_Experiment"));
        QIcon icon14;
        icon14.addFile(QString::fromUtf8(":/icons/viewold.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionView_Experiment->setIcon(icon14);
        actionQuick_Experiment = new QAction(MainWindow);
        actionQuick_Experiment->setObjectName(QString::fromUtf8("actionQuick_Experiment"));
        QIcon icon15;
        icon15.addFile(QString::fromUtf8(":/icons/quickexpt.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionQuick_Experiment->setIcon(icon15);
        actionStart_Sequence = new QAction(MainWindow);
        actionStart_Sequence->setObjectName(QString::fromUtf8("actionStart_Sequence"));
        QIcon icon16;
        icon16.addFile(QString::fromUtf8(":/icons/sequence.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionStart_Sequence->setIcon(icon16);
#ifdef BC_LIF
        QIcon lifIcon;
        lifIcon.addFile(QString(":/icons/laser.png"),QSize(), QIcon::Normal, QIcon::Off);
        actionLifConfig = new QAction("LIF Configuration",MainWindow);
        actionLifConfig->setObjectName("ActionLifConfig");
        actionLifConfig->setIcon(lifIcon);
#endif

        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        mainLayout = new QHBoxLayout(centralWidget);
        mainLayout->setSpacing(6);
        mainLayout->setContentsMargins(11, 11, 11, 11);
        mainLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        instrumentStatusLayout = new QVBoxLayout();
        instrumentStatusLayout->setSpacing(6);
        instrumentStatusLayout->setObjectName(QString::fromUtf8("instrumentStatusLayout"));
        instStatusLabel = new QLabel(centralWidget);
        instStatusLabel->setObjectName(QString::fromUtf8("label"));

        acquireButton = new QToolButton(MainWindow);
        acquireButton->setText("Acquire");
        acquireButton->setToolTip("Start a new acquisition");
        acquireButton->setIcon(icon1);
        acquireButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        acquireButton->setPopupMode(QToolButton::InstantPopup);

        QIcon hwIcon;
        hwIcon.addFile(QString(":/icons/bc.png"),QSize(), QIcon::Normal, QIcon::Off);
        hardwareButton = new QToolButton(MainWindow);
        hardwareButton->setText("Hardware");
        hardwareButton->setToolTip("Configure hardware settings");
        hardwareButton->setIcon(hwIcon);
        hardwareButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        hardwareButton->setPopupMode(QToolButton::InstantPopup);

        instStatusLabel->setAlignment(Qt::AlignCenter);

        auxPlotButton = new QToolButton(MainWindow);
        auxPlotButton->setText(QString("Aux Data"));
        auxPlotButton->setToolTip("Configure settings for aux data plots");
        auxPlotButton->setIcon(auxIcon);
        auxPlotButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        auxPlotButton->setPopupMode(QToolButton::InstantPopup);

        rollingPlotButton = new QToolButton(MainWindow);
        rollingPlotButton->setText(QString("Rolling Data"));
        rollingPlotButton->setToolTip("Configure settings for rolling data plots");
        rollingPlotButton->setIcon(rollIcon);
        rollingPlotButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        rollingPlotButton->setPopupMode(QToolButton::InstantPopup);

        QIcon settingsIcon;
        settingsIcon.addFile(QString(":/icons/menu.svg"),QSize(), QIcon::Normal, QIcon::Off);
        settingsButton = new QToolButton(MainWindow);
        settingsButton->setIcon(settingsIcon);
        settingsButton->setText("Settings");
        settingsButton->setToolTip("Configure application and miscellaneous settings");
        settingsButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        settingsButton->setPopupMode(QToolButton::InstantPopup);

        QIcon fontIcon;
        fontIcon.addFile(QString(":/icons/font.svg"),QSize(), QIcon::Normal, QIcon::Off);
        fontAction = new QAction("Application Font");
        fontAction->setIcon(fontIcon);

        QIcon saveIcon;
        saveIcon.addFile(QString(":/icons/save-as.svg"),QSize(), QIcon::Normal, QIcon::Off);
        savePathAction = new QAction("Data Storage");
        savePathAction->setIcon(saveIcon);


        instrumentStatusLayout->addWidget(instStatusLabel);

        statusLayout = new QGridLayout();
        statusLayout->setSpacing(6);
        statusLayout->setObjectName(QString::fromUtf8("statusLayout"));

        exptLabel = new QLabel(centralWidget);
        exptLabel->setObjectName(QString::fromUtf8("exptLabel"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(exptLabel->sizePolicy().hasHeightForWidth());
        exptLabel->setSizePolicy(sizePolicy);
        exptLabel->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        statusLayout->addWidget(exptLabel,0,0);

        exptSpinBox = new QSpinBox(centralWidget);
        exptSpinBox->setObjectName(QString::fromUtf8("exptSpinBox"));
        QSizePolicy sizePolicy1(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(exptSpinBox->sizePolicy().hasHeightForWidth());
        exptSpinBox->setSizePolicy(sizePolicy1);
        exptSpinBox->setFocusPolicy(Qt::ClickFocus);
        exptSpinBox->setReadOnly(true);
        exptSpinBox->setButtonSymbols(QAbstractSpinBox::NoButtons);
        exptSpinBox->setMaximum(2147483647);
        exptSpinBox->blockSignals(true);

        statusLayout->addWidget(exptSpinBox,0,1);



        instrumentStatusLayout->addLayout(statusLayout);

        clockBox = new ClockDisplayBox(centralWidget);
        instrumentStatusLayout->addWidget(clockBox,0);

        statusSpacer = new QSpacerItem(0, 0, QSizePolicy::Minimum, QSizePolicy::Expanding);

        instrumentStatusLayout->addItem(statusSpacer);

        ftmwProgressLabel = new QLabel(centralWidget);
        ftmwProgressLabel->setObjectName(QString::fromUtf8("label_2"));
        ftmwProgressLabel->setAlignment(Qt::AlignCenter);

        instrumentStatusLayout->addWidget(ftmwProgressLabel);

        ftmwProgressBar = new QProgressBar(centralWidget);
        ftmwProgressBar->setObjectName(QString::fromUtf8("ftmwProgressBar"));
        ftmwProgressBar->setValue(0);

        instrumentStatusLayout->addWidget(ftmwProgressBar);

        mainLayout->addLayout(instrumentStatusLayout);

        mainTabWidget = new QTabWidget(centralWidget);
        mainTabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        mainTabWidget->setTabPosition(QTabWidget::East);

        ftmwTab = new QWidget();
        ftmwTab->setObjectName(QString::fromUtf8("ftmwTab"));
        ftmwTabLayout = new QVBoxLayout(ftmwTab);
        ftmwTabLayout->setSpacing(6);
        ftmwTabLayout->setContentsMargins(11, 11, 11, 11);
        ftmwTabLayout->setObjectName(QString::fromUtf8("verticalLayout_3"));
        ftViewWidget = new FtmwViewWidget(ftmwTab);
        ftViewWidget->setObjectName(QString::fromUtf8("ftViewWidget"));

        ftmwTabLayout->addWidget(ftViewWidget);

        mainTabWidget->addTab(ftmwTab, rfIcon, QString());

#ifdef BC_LIF
        lifTab = new QWidget();
        lifTab->setObjectName(QString("lifTab"));
        lifTabLayout = new QVBoxLayout(lifTab);
        lifDisplayWidget = new LifDisplayWidget();
        lifDisplayWidget->setObjectName(QString("lifDisplayWidget"));
        lifTabLayout->addWidget(lifDisplayWidget);
        mainTabWidget->addTab(lifTab,lifIcon,QString());
        mainTabWidget->setTabText(mainTabWidget->indexOf(lifTab),QString("LIF"));

        lifProgressBar = new QProgressBar();
        instrumentStatusLayout->addWidget(new QLabel(QString("LIF Progress")),0,Qt::AlignCenter);
        instrumentStatusLayout->addWidget(lifProgressBar);
#endif

        rollingDataTab = new QWidget();
        rollingDataTab->setObjectName(QString::fromUtf8("rollingDataTab"));
        rollingDataViewLayout = new QVBoxLayout(rollingDataTab);
        rollingDataViewLayout->setSpacing(6);
        rollingDataViewLayout->setContentsMargins(11, 11, 11, 11);
        rollingDataViewLayout->setObjectName(QString::fromUtf8("rollingDataViewLayout"));
        rollingDataViewWidget = new RollingDataWidget(BC::Key::rollingDataWidget,rollingDataTab);
        rollingDataViewWidget->setObjectName(QString::fromUtf8("rollingDataViewWidget"));

        rollingDataViewLayout->addWidget(rollingDataViewWidget);

        mainTabWidget->addTab(rollingDataTab, rollIcon, QString());


        auxDataTab = new QWidget();
        auxDataTab->setObjectName(QString::fromUtf8("auxDataTab"));
        auxDataViewLayout = new QVBoxLayout(auxDataTab);
        auxDataViewLayout->setSpacing(6);
        auxDataViewLayout->setContentsMargins(11, 11, 11, 11);
        auxDataViewLayout->setObjectName(QString::fromUtf8("auxDataViewLayout"));
        auxDataViewWidget = new AuxDataViewWidget(BC::Key::auxDataWidget,auxDataTab);
        auxDataViewWidget->setObjectName(QString::fromUtf8("auxDataViewWidget"));

        auxDataViewLayout->addWidget(auxDataViewWidget);

        mainTabWidget->addTab(auxDataTab, auxIcon, QString());



        logTab = new QWidget();
        logTab->setObjectName(QString::fromUtf8("logTab"));
        logTabLayout = new QVBoxLayout(logTab);
        logTabLayout->setSpacing(6);
        logTabLayout->setContentsMargins(11, 11, 11, 11);
        logTabLayout->setObjectName(QString::fromUtf8("verticalLayout_2"));
        logTextEdit = new QTextEdit(logTab);
        logTextEdit->setObjectName(QString::fromUtf8("log"));
        logTextEdit->setUndoRedoEnabled(false);
        logTextEdit->setReadOnly(true);

        logTabLayout->addWidget(logTextEdit);

        mainTabWidget->addTab(logTab, QString());

        mainLayout->addWidget(mainTabWidget);

        mainLayout->setStretch(1, 1);
        MainWindow->setCentralWidget(centralWidget);
//        menuBar = new QMenuBar(centralWidget);
//        menuBar->setObjectName(QString::fromUtf8("menuBar"));
//        menuBar->setGeometry(QRect(0, 0, 676, 23));
        menuHardware = new QMenu(hardwareButton);
        menuHardware->setObjectName(QString::fromUtf8("menuHardware"));
        menuAcquisition = new QMenu(acquireButton);
        menuAcquisition->setObjectName(QString::fromUtf8("menuAcquisition"));

        menuAuxData = new QMenu(auxPlotButton);
        menuAuxData->setObjectName(QString::fromUtf8("menuAuxData"));
        menuAuxData->setIcon(auxIcon);
        menuRollingData = new QMenu(rollingPlotButton);
        menuRollingData->setObjectName(QString::fromUtf8("menuRollingData"));
        menuRollingData->setIcon(rollIcon);
//        MainWindow->setMenuBar(menuBar);

        settingsMenu = new QMenu(settingsButton);
        settingsMenu->addAction(fontAction);
        settingsMenu->addAction(savePathAction);

        mainToolBar = new QToolBar(centralWidget);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        mainToolBar->setIconSize(QSize(14, 14));
        mainToolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        mainToolBar->setMovable(false);
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(centralWidget);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        MainWindow->setStatusBar(statusBar);

        rollingDurationBox = new SpinBoxWidgetAction("History",menuRollingData);
        rollingDurationBox->setRange(1,48);
        rollingDurationBox->setSuffix(" hr");

        auxGraphsBox = new SpinBoxWidgetAction("Graphs",menuAuxData);
        auxGraphsBox->setObjectName(QString::fromUtf8("auxGraphsBox"));
        auxGraphsBox->setIcon(icon7);
        auxGraphsBox->setRange(1,9);

        rollingGraphsBox = new SpinBoxWidgetAction("Graphs",menuRollingData);
        rollingGraphsBox->setObjectName(QString::fromUtf8("rollingGraphsBox"));
        rollingGraphsBox->setIcon(icon7);
        rollingGraphsBox->setRange(1,9);

        menuHardware->addAction(actionCommunication);
        menuHardware->addAction(actionTest_All_Connections);
        menuHardware->addSeparator();
        menuHardware->addAction(actionRfConfig);
#ifdef BC_LIF
        menuHardware->addAction(actionLifConfig);
#endif
        menuHardware->addSeparator();
        menuAcquisition->addAction(actionStart_Experiment);
        menuAcquisition->addAction(actionQuick_Experiment);
        menuAcquisition->addAction(actionStart_Sequence);
        menuAcquisition->addSeparator();
        menuRollingData->addAction(actionAutoscale_Rolling);
        menuRollingData->addAction(rollingGraphsBox);
        menuRollingData->addAction(rollingDurationBox);
        menuAuxData->addAction(actionAutoscale_Aux);
        menuAuxData->addAction(auxGraphsBox);

        mainToolBar->addWidget(acquireButton);
        acquireButton->setMenu(menuAcquisition);

        mainToolBar->addWidget(hardwareButton);
        hardwareButton->setMenu(menuHardware);

        mainToolBar->addWidget(rollingPlotButton);
        rollingPlotButton->setMenu(menuRollingData);

        mainToolBar->addWidget(auxPlotButton);
        auxPlotButton->setMenu(menuAuxData);

        mainToolBar->addAction(actionView_Experiment);

        mainToolBar->addWidget(settingsButton);
        settingsButton->setMenu(settingsMenu);

        auto w = new QWidget(MainWindow);
        w->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Fixed);
        mainToolBar->addWidget(w);
        mainToolBar->addSeparator();
        mainToolBar->addWidget(pauseButton);
        mainToolBar->addWidget(resumeButton);
        mainToolBar->addWidget(abortButton);
        mainToolBar->addWidget(sleepButton);

        retranslateUi(MainWindow);

        mainTabWidget->setCurrentIndex(0);

        for(int i=0; i<mainTabWidget->count(); i++)
        {
            mainTabWidget->widget(i)->layout()->setContentsMargins(0,0,0,0);
            mainTabWidget->widget(i)->layout()->setMargin(0);
        }


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Blackchirp", nullptr));
        actionStart_Experiment->setText(QApplication::translate("MainWindow", "&Start Experiment", nullptr));
#ifndef QT_NO_SHORTCUT
        actionStart_Experiment->setShortcut(QApplication::translate("MainWindow", "F2", nullptr));
#endif // QT_NO_SHORTCUT
        pauseButton->setText(QApplication::translate("MainWindow", "&Pause", nullptr));
#ifndef QT_NO_SHORTCUT
        pauseButton->setShortcut(QApplication::translate("MainWindow", "F4", nullptr));
#endif // QT_NO_SHORTCUT
        resumeButton->setText(QApplication::translate("MainWindow", "&Resume", nullptr));
#ifndef QT_NO_SHORTCUT
        resumeButton->setShortcut(QApplication::translate("MainWindow", "F5", nullptr));
#endif // QT_NO_SHORTCUT
        abortButton->setText(QApplication::translate("MainWindow", "&Abort", nullptr));
#ifndef QT_NO_SHORTCUT
        abortButton->setShortcut(QApplication::translate("MainWindow", "F6", nullptr));
#endif // QT_NO_SHORTCUT
        actionCommunication->setText(QApplication::translate("MainWindow", "&Communication", nullptr));
#ifndef QT_NO_SHORTCUT
        actionCommunication->setShortcut(QApplication::translate("MainWindow", "Ctrl+H", nullptr));
#endif // QT_NO_SHORTCUT
        auxGraphsBox->setText(QApplication::translate("MainWindow", "# &Graphs...", nullptr));
        rollingGraphsBox->setText(QApplication::translate("MainWindow", "# &Graphs...", nullptr));
        actionTest_All_Connections->setText(QApplication::translate("MainWindow", "&Test All Connections", nullptr));
#ifndef QT_NO_SHORTCUT
        actionTest_All_Connections->setShortcut(QApplication::translate("MainWindow", "Ctrl+T", nullptr));
#endif // QT_NO_SHORTCUT
        sleepButton->setText(QApplication::translate("MainWindow", "&Sleep", nullptr));
#ifndef QT_NO_SHORTCUT
        sleepButton->setShortcut(QApplication::translate("MainWindow", "F8", nullptr));
#endif // QT_NO_SHORTCUT
        actionAutoscale_Rolling->setText(QApplication::translate("MainWindow", "&Autoscale All", nullptr));
        actionAutoscale_Aux->setText(QApplication::translate("MainWindow", "&Autoscale All", nullptr));
        actionView_Experiment->setText(QApplication::translate("MainWindow", "&View Experiment...", nullptr));
        actionQuick_Experiment->setText(QApplication::translate("MainWindow", "&Quick Experiment", nullptr));
#ifndef QT_NO_SHORTCUT
        actionQuick_Experiment->setShortcut(QApplication::translate("MainWindow", "F3", nullptr));
#endif // QT_NO_SHORTCUT
        actionStart_Sequence->setText(QApplication::translate("MainWindow", "Start Seq&uence", nullptr));
        instStatusLabel->setText(QApplication::translate("MainWindow", "Instrument Status", nullptr));
        exptLabel->setText(QApplication::translate("MainWindow", "Expt", nullptr));
#ifndef QT_NO_TOOLTIP
        exptSpinBox->setToolTip(QApplication::translate("MainWindow", "Number of the most recent experiment", nullptr));
#endif // QT_NO_TOOLTIP
        ftmwProgressLabel->setText(QApplication::translate("MainWindow", "FTMW Progress", nullptr));
        mainTabWidget->setTabText(mainTabWidget->indexOf(ftmwTab), QApplication::translate("MainWindow", "CP-FTMW", nullptr));
        mainTabWidget->setTabText(mainTabWidget->indexOf(rollingDataTab), QApplication::translate("MainWindow", "Rolling Data", nullptr));
        mainTabWidget->setTabText(mainTabWidget->indexOf(auxDataTab), QApplication::translate("MainWindow", "Aux Data", nullptr));
        mainTabWidget->setTabText(mainTabWidget->indexOf(logTab), QApplication::translate("MainWindow", "Log", nullptr));

        menuHardware->setTitle(QApplication::translate("MainWindow", "Hardware", nullptr));
        menuAcquisition->setTitle(QApplication::translate("MainWindow", "Acquisition", nullptr));
        menuRollingData->setTitle(QApplication::translate("MainWindow", "Rolling Data", nullptr));
        menuAuxData->setTitle(QApplication::translate("MainWindow", "Aux Data", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui


#endif // MAINWINDOW_UI_H
