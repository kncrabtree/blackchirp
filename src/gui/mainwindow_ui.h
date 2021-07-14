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
#include <gui/widget/ftmwviewwidget.h>
#include <gui/widget/led.h>
#include <gui/widget/pulseconfigwidget.h>
#include <gui/widget/auxdataviewwidget.h>
#include <gui/widget/gascontrolwidget.h>
#include <gui/widget/gasflowdisplaywidget.h>

class Ui_MainWindow
{
public:
    QAction *actionStart_Experiment;
    QAction *actionPause;
    QAction *actionResume;
    QAction *actionAbort;
    QAction *actionCommunication;
    QAction *actionAuxDataShow;
    QAction *actionRollingDataShow;
    QAction *action_AuxGraphs;
    QAction *action_RollingGraphs;
    QAction *actionTest_All_Connections;
    QAction *actionSleep;
    QAction *actionAutoscale_Rolling;
    QAction *actionAutoscale_Aux;
    QAction *actionCP_FTMW;
    QAction *actionControl;
    QAction *actionLog;
    QAction *actionView_Experiment;
    QAction *actionQuick_Experiment;
    QAction *actionStart_Sequence;
    QWidget *centralWidget;
    QHBoxLayout *mainLayout;
    QVBoxLayout *instrumentStatusLayout;
    QLabel *instStatusLabel;
    QFormLayout *statusLayout;
    QLabel *exptLabel;
    QSpinBox *exptSpinBox;
    QGroupBox *flowStatusBox;
    GasFlowDisplayWidget *gasFlowDisplayWidget;
    QGroupBox *pulseConfigBox;
    QSpacerItem *statusSpacer;
    QLabel *ftmwProgressLabel;
    QProgressBar *ftmwProgressBar;
    QTabWidget *mainTabWidget;
    QWidget *controlTab;
    QVBoxLayout *controlTabLayout;
    QHBoxLayout *controlTabTopLayout;
    QVBoxLayout *gasControlLayout;
    QGroupBox *gasControlBox;
    GasControlWidget *gasControlWidget;
    PulseConfigWidget *pulseConfigWidget;
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
    QMenuBar *menuBar;
    QMenu *menuHardware;
    QMenu *menuAcquisition;
    QMenu *menuView;
    QMenu *menuRollingData;
    QMenu *menuAuxData;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
//        MainWindow->resize(676, 309);
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/bc_logo_small.png"), QSize(), QIcon::Normal, QIcon::Off);
        MainWindow->setWindowIcon(icon);
        actionStart_Experiment = new QAction(MainWindow);
        actionStart_Experiment->setObjectName(QString::fromUtf8("actionStart_Experiment"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icons/experiment.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionStart_Experiment->setIcon(icon1);
        actionPause = new QAction(MainWindow);
        actionPause->setObjectName(QString::fromUtf8("actionPause"));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/icons/pause.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionPause->setIcon(icon2);
        actionResume = new QAction(MainWindow);
        actionResume->setObjectName(QString::fromUtf8("actionResume"));
        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/icons/start.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionResume->setIcon(icon3);
        actionAbort = new QAction(MainWindow);
        actionAbort->setObjectName(QString::fromUtf8("actionAbort"));
        QIcon icon4;
        icon4.addFile(QString::fromUtf8(":/icons/abort.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionAbort->setIcon(icon4);
        actionCommunication = new QAction(MainWindow);
        actionCommunication->setObjectName(QString::fromUtf8("actionCommunication"));
        QIcon icon5;
        icon5.addFile(QString::fromUtf8(":/icons/computer.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionCommunication->setIcon(icon5);
        actionAuxDataShow = new QAction(MainWindow);
        actionAuxDataShow->setObjectName(QString::fromUtf8("actionAuxDataShow"));
        QIcon icon6;
        icon6.addFile(QString::fromUtf8(":/icons/dataplots.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionAuxDataShow->setIcon(icon6);
        actionRollingDataShow = new QAction(MainWindow);
        actionRollingDataShow->setObjectName(QString::fromUtf8("actionRollingDataShow"));
        actionRollingDataShow->setIcon(icon6);
        action_AuxGraphs = new QAction(MainWindow);
        action_AuxGraphs->setObjectName(QString::fromUtf8("action_AuxGraphs"));
        QIcon icon7;
        icon7.addFile(QString::fromUtf8(":/icons/num.png"), QSize(), QIcon::Normal, QIcon::Off);
        action_AuxGraphs->setIcon(icon7);
        action_RollingGraphs = new QAction(MainWindow);
        action_RollingGraphs->setObjectName(QString::fromUtf8("action_RollingGraphs"));
        action_RollingGraphs->setIcon(icon7);
        actionTest_All_Connections = new QAction(MainWindow);
        actionTest_All_Connections->setObjectName(QString::fromUtf8("actionTest_All_Connections"));
        QIcon icon8;
        icon8.addFile(QString::fromUtf8(":/icons/connect.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionTest_All_Connections->setIcon(icon8);
        actionSleep = new QAction(MainWindow);
        actionSleep->setObjectName(QString::fromUtf8("actionSleep"));
        actionSleep->setCheckable(true);
        QIcon icon9;
        icon9.addFile(QString::fromUtf8(":/icons/sleep.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionSleep->setIcon(icon9);
        actionAutoscale_Aux = new QAction(MainWindow);
        actionAutoscale_Aux->setObjectName(QString::fromUtf8("actionAutoscale_Aux"));
        QIcon icon10;
        icon10.addFile(QString::fromUtf8(":/icons/autoscale.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionAutoscale_Aux->setIcon(icon10);
        actionAutoscale_Rolling = new QAction(MainWindow);
        actionAutoscale_Rolling->setObjectName(QString::fromUtf8("actionAutoscale_Rolling"));
        actionAutoscale_Rolling->setIcon(icon10);
        actionCP_FTMW = new QAction(MainWindow);
        actionCP_FTMW->setObjectName(QString::fromUtf8("actionCP_FTMW"));
        QIcon icon11;
        icon11.addFile(QString::fromUtf8(":/icons/chirp.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionCP_FTMW->setIcon(icon11);
        actionControl = new QAction(MainWindow);
        actionControl->setObjectName(QString::fromUtf8("actionControl"));
        QIcon icon12;
        icon12.addFile(QString::fromUtf8(":/icons/controltab.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionControl->setIcon(icon12);
        actionLog = new QAction(MainWindow);
        actionLog->setObjectName(QString::fromUtf8("actionLog"));
        QIcon icon13;
        icon13.addFile(QString::fromUtf8(":/icons/log.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionLog->setIcon(icon13);
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
        QFont font;
        font.setPointSize(8);
        instStatusLabel->setFont(font);
        instStatusLabel->setAlignment(Qt::AlignCenter);

        instrumentStatusLayout->addWidget(instStatusLabel);

        statusLayout = new QFormLayout();
        statusLayout->setSpacing(6);
        statusLayout->setObjectName(QString::fromUtf8("formLayout"));
        statusLayout->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);
        statusLayout->setFormAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);
        exptLabel = new QLabel(centralWidget);
        exptLabel->setObjectName(QString::fromUtf8("exptLabel"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::MinimumExpanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(exptLabel->sizePolicy().hasHeightForWidth());
        exptLabel->setSizePolicy(sizePolicy);
        exptLabel->setFont(font);
        exptLabel->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        statusLayout->setWidget(0, QFormLayout::LabelRole, exptLabel);

        exptSpinBox = new QSpinBox(centralWidget);
        exptSpinBox->setObjectName(QString::fromUtf8("exptSpinBox"));
        QSizePolicy sizePolicy1(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(exptSpinBox->sizePolicy().hasHeightForWidth());
        exptSpinBox->setSizePolicy(sizePolicy1);
        exptSpinBox->setFont(font);
        exptSpinBox->setFocusPolicy(Qt::ClickFocus);
        exptSpinBox->setReadOnly(true);
        exptSpinBox->setButtonSymbols(QAbstractSpinBox::NoButtons);
        exptSpinBox->setMaximum(2147483647);

        statusLayout->setWidget(0, QFormLayout::FieldRole, exptSpinBox);


        instrumentStatusLayout->addLayout(statusLayout);

        flowStatusBox = new QGroupBox(centralWidget);
        flowStatusBox->setObjectName(QString::fromUtf8("flowStatusBox"));
        flowStatusBox->setFont(font);

        gasFlowDisplayWidget = new GasFlowDisplayWidget;
        flowStatusBox->setLayout(gasFlowDisplayWidget->layout());


        instrumentStatusLayout->addWidget(flowStatusBox);

        pulseConfigBox = new QGroupBox(centralWidget);
        pulseConfigBox->setObjectName(QString::fromUtf8("pulseConfigBox"));
        pulseConfigBox->setFont(font);

        instrumentStatusLayout->addWidget(pulseConfigBox);

        statusSpacer = new QSpacerItem(0, 0, QSizePolicy::Minimum, QSizePolicy::Expanding);

        instrumentStatusLayout->addItem(statusSpacer);

        ftmwProgressLabel = new QLabel(centralWidget);
        ftmwProgressLabel->setObjectName(QString::fromUtf8("label_2"));
        ftmwProgressLabel->setFont(font);
        ftmwProgressLabel->setAlignment(Qt::AlignCenter);

        instrumentStatusLayout->addWidget(ftmwProgressLabel);

        ftmwProgressBar = new QProgressBar(centralWidget);
        ftmwProgressBar->setObjectName(QString::fromUtf8("ftmwProgressBar"));
        ftmwProgressBar->setFont(font);
        ftmwProgressBar->setValue(0);

        instrumentStatusLayout->addWidget(ftmwProgressBar);

        instrumentStatusLayout->setStretch(4, 1);

        mainLayout->addLayout(instrumentStatusLayout);

        mainTabWidget = new QTabWidget(centralWidget);
        mainTabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        mainTabWidget->setTabPosition(QTabWidget::East);
        controlTab = new QWidget();
        controlTab->setObjectName(QString::fromUtf8("controlTab"));
        controlTabLayout = new QVBoxLayout(controlTab);
        controlTabLayout->setSpacing(6);
        controlTabLayout->setContentsMargins(11, 11, 11, 11);
        controlTabLayout->setObjectName(QString::fromUtf8("verticalLayout_5"));
        controlTabTopLayout = new QHBoxLayout();
        controlTabTopLayout->setSpacing(6);
        controlTabTopLayout->setObjectName(QString::fromUtf8("controlTopLayout"));
        gasControlLayout = new QVBoxLayout();
        gasControlLayout->setSpacing(6);
        gasControlLayout->setObjectName(QString::fromUtf8("gasControlLayout"));

        gasControlBox = new QGroupBox;
        gasControlBox->setObjectName(QString::fromUtf8("gasControlBox"));

        gasControlWidget = new GasControlWidget(centralWidget);
        gasControlBox->setLayout(gasControlWidget->layout());
        gasFlowDisplayWidget->initChannelNames(gasControlWidget->getGasNames());


        gasControlLayout->addWidget(gasControlBox);

        controlTabTopLayout->addLayout(gasControlLayout);


        controlTabLayout->addLayout(controlTabTopLayout);

        pulseConfigWidget = new PulseConfigWidget(controlTab);
        pulseConfigWidget->setObjectName(QString::fromUtf8("pulseConfigWidget"));

        controlTabLayout->addWidget(pulseConfigWidget);

        controlTabLayout->setStretch(0, 1);
        controlTabLayout->setStretch(1, 1);
        mainTabWidget->addTab(controlTab, icon12, QString());
        ftmwTab = new QWidget();
        ftmwTab->setObjectName(QString::fromUtf8("ftmwTab"));
        ftmwTabLayout = new QVBoxLayout(ftmwTab);
        ftmwTabLayout->setSpacing(6);
        ftmwTabLayout->setContentsMargins(11, 11, 11, 11);
        ftmwTabLayout->setObjectName(QString::fromUtf8("verticalLayout_3"));
        ftViewWidget = new FtmwViewWidget(ftmwTab);
        ftViewWidget->setObjectName(QString::fromUtf8("ftViewWidget"));

        ftmwTabLayout->addWidget(ftViewWidget);

        mainTabWidget->addTab(ftmwTab, icon11, QString());


        rollingDataTab = new QWidget();
        rollingDataTab->setObjectName(QString::fromUtf8("rollingDataTab"));
        rollingDataViewLayout = new QVBoxLayout(rollingDataTab);
        rollingDataViewLayout->setSpacing(6);
        rollingDataViewLayout->setContentsMargins(11, 11, 11, 11);
        rollingDataViewLayout->setObjectName(QString::fromUtf8("rollingDataViewLayout"));
        rollingDataViewWidget = new RollingDataWidget(BC::Key::rollingDataWidget,rollingDataTab);
        rollingDataViewWidget->setObjectName(QString::fromUtf8("rollingDataViewWidget"));

        rollingDataViewLayout->addWidget(rollingDataViewWidget);

        mainTabWidget->addTab(rollingDataTab, icon6, QString());


        auxDataTab = new QWidget();
        auxDataTab->setObjectName(QString::fromUtf8("auxDataTab"));
        auxDataViewLayout = new QVBoxLayout(auxDataTab);
        auxDataViewLayout->setSpacing(6);
        auxDataViewLayout->setContentsMargins(11, 11, 11, 11);
        auxDataViewLayout->setObjectName(QString::fromUtf8("auxDataViewLayout"));
        auxDataViewWidget = new AuxDataViewWidget(BC::Key::auxDataWidget,auxDataTab);
        auxDataViewWidget->setObjectName(QString::fromUtf8("auxDataViewWidget"));

        auxDataViewLayout->addWidget(auxDataViewWidget);

        mainTabWidget->addTab(auxDataTab, icon6, QString());



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
        menuBar = new QMenuBar(centralWidget);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 676, 23));
        menuHardware = new QMenu(menuBar);
        menuHardware->setObjectName(QString::fromUtf8("menuHardware"));
        menuAcquisition = new QMenu(menuBar);
        menuAcquisition->setObjectName(QString::fromUtf8("menuAcquisition"));
        menuView = new QMenu(menuBar);
        menuView->setObjectName(QString::fromUtf8("menuView"));
        menuAuxData = new QMenu(menuView);
        menuAuxData->setObjectName(QString::fromUtf8("menuAuxData"));
        menuAuxData->setIcon(icon6);
        menuRollingData = new QMenu(menuView);
        menuRollingData->setObjectName(QString::fromUtf8("menuRollingData"));
        menuRollingData->setIcon(icon6);
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(centralWidget);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        mainToolBar->setIconSize(QSize(14, 14));
        mainToolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(centralWidget);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        MainWindow->setStatusBar(statusBar);

        menuBar->addAction(menuAcquisition->menuAction());
        menuBar->addAction(menuHardware->menuAction());
        menuBar->addAction(menuView->menuAction());
        menuHardware->addAction(actionSleep);
        menuHardware->addSeparator();
        menuHardware->addAction(actionCommunication);
        menuHardware->addAction(actionTest_All_Connections);
        menuAcquisition->addAction(actionStart_Experiment);
        menuAcquisition->addAction(actionQuick_Experiment);
        menuAcquisition->addAction(actionStart_Sequence);
        menuAcquisition->addAction(actionPause);
        menuAcquisition->addAction(actionResume);
        menuAcquisition->addAction(actionAbort);
        menuAcquisition->addSeparator();
        menuView->addAction(actionControl);
        menuView->addAction(actionCP_FTMW);
        menuView->addAction(menuRollingData->menuAction());
        menuView->addAction(menuAuxData->menuAction());
        menuView->addAction(actionLog);
        menuView->addSeparator();
        menuView->addAction(actionView_Experiment);
        menuRollingData->addAction(actionRollingDataShow);
        menuRollingData->addAction(actionAutoscale_Rolling);
        menuRollingData->addAction(action_RollingGraphs);
        menuAuxData->addAction(actionAuxDataShow);
        menuAuxData->addAction(actionAutoscale_Aux);
        menuAuxData->addAction(action_AuxGraphs);
        mainToolBar->addAction(actionStart_Experiment);
        mainToolBar->addAction(actionQuick_Experiment);
        mainToolBar->addAction(actionPause);
        mainToolBar->addAction(actionResume);
        mainToolBar->addAction(actionAbort);
        mainToolBar->addAction(actionSleep);
        mainToolBar->addAction(actionView_Experiment);

        retranslateUi(MainWindow);

        mainTabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "BlackChirp", nullptr));
        actionStart_Experiment->setText(QApplication::translate("MainWindow", "&Start Experiment", nullptr));
#ifndef QT_NO_SHORTCUT
        actionStart_Experiment->setShortcut(QApplication::translate("MainWindow", "F2", nullptr));
#endif // QT_NO_SHORTCUT
        actionPause->setText(QApplication::translate("MainWindow", "&Pause", nullptr));
#ifndef QT_NO_SHORTCUT
        actionPause->setShortcut(QApplication::translate("MainWindow", "F4", nullptr));
#endif // QT_NO_SHORTCUT
        actionResume->setText(QApplication::translate("MainWindow", "&Resume", nullptr));
#ifndef QT_NO_SHORTCUT
        actionResume->setShortcut(QApplication::translate("MainWindow", "F5", nullptr));
#endif // QT_NO_SHORTCUT
        actionAbort->setText(QApplication::translate("MainWindow", "&Abort", nullptr));
#ifndef QT_NO_SHORTCUT
        actionAbort->setShortcut(QApplication::translate("MainWindow", "F6", nullptr));
#endif // QT_NO_SHORTCUT
        actionCommunication->setText(QApplication::translate("MainWindow", "&Communication", nullptr));
#ifndef QT_NO_SHORTCUT
        actionCommunication->setShortcut(QApplication::translate("MainWindow", "Ctrl+H", nullptr));
#endif // QT_NO_SHORTCUT
        actionAuxDataShow->setText(QApplication::translate("MainWindow", "&Show", nullptr));
        actionRollingDataShow->setText(QApplication::translate("MainWindow", "&Show", nullptr));
        action_AuxGraphs->setText(QApplication::translate("MainWindow", "# &Graphs...", nullptr));
        action_RollingGraphs->setText(QApplication::translate("MainWindow", "# &Graphs...", nullptr));
        actionTest_All_Connections->setText(QApplication::translate("MainWindow", "&Test All Connections", nullptr));
#ifndef QT_NO_SHORTCUT
        actionTest_All_Connections->setShortcut(QApplication::translate("MainWindow", "Ctrl+T", nullptr));
#endif // QT_NO_SHORTCUT
        actionSleep->setText(QApplication::translate("MainWindow", "&Sleep", nullptr));
#ifndef QT_NO_SHORTCUT
        actionSleep->setShortcut(QApplication::translate("MainWindow", "F8", nullptr));
#endif // QT_NO_SHORTCUT
        actionAutoscale_Rolling->setText(QApplication::translate("MainWindow", "&Autoscale All", nullptr));
        actionAutoscale_Aux->setText(QApplication::translate("MainWindow", "&Autoscale All", nullptr));
        actionCP_FTMW->setText(QApplication::translate("MainWindow", "&CP-FTMW", nullptr));
        actionControl->setText(QApplication::translate("MainWindow", "C&ontrol", nullptr));
        actionLog->setText(QApplication::translate("MainWindow", "Lo&g", nullptr));
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
        flowStatusBox->setTitle(QApplication::translate("MainWindow", "Flow Status", nullptr));
        pulseConfigBox->setTitle(QApplication::translate("MainWindow", "Pulse Configuration", nullptr));
        ftmwProgressLabel->setText(QApplication::translate("MainWindow", "FTMW Progress", nullptr));
        gasControlBox->setTitle(QApplication::translate("MainWindow", "Gas Control", nullptr));
        mainTabWidget->setTabText(mainTabWidget->indexOf(controlTab), QApplication::translate("MainWindow", "Control", nullptr));
        mainTabWidget->setTabText(mainTabWidget->indexOf(ftmwTab), QApplication::translate("MainWindow", "CP-FTMW", nullptr));
        mainTabWidget->setTabText(mainTabWidget->indexOf(rollingDataTab), QApplication::translate("MainWindow", "Rolling Data", nullptr));
        mainTabWidget->setTabText(mainTabWidget->indexOf(auxDataTab), QApplication::translate("MainWindow", "Aux Data", nullptr));
        mainTabWidget->setTabText(mainTabWidget->indexOf(logTab), QApplication::translate("MainWindow", "Log", nullptr));
        menuHardware->setTitle(QApplication::translate("MainWindow", "Ha&rdware", nullptr));
        menuAcquisition->setTitle(QApplication::translate("MainWindow", "Ac&quisition", nullptr));
        menuView->setTitle(QApplication::translate("MainWindow", "&View", nullptr));
        menuRollingData->setTitle(QApplication::translate("MainWindow", "&Rolling Data", nullptr));
        menuAuxData->setTitle(QApplication::translate("MainWindow", "&Aux Data", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui


#endif // MAINWINDOW_UI_H
