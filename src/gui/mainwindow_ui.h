#ifndef MAINWINDOW_UI_H
#define MAINWINDOW_UI_H


#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QAction>
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
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QFrame>
#include <gui/widget/ftmwviewwidget.h>
#include <gui/widget/led.h>
#include <gui/widget/auxdataviewwidget.h>
#include <gui/widget/clockdisplaybox.h>
#include <gui/widget/toolbarwidgetaction.h>
#include <gui/style/themecolors.h>

#include <gui/lif/gui/lifdisplaywidget.h>
#include <data/storage/applicationconfigmanager.h>

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
    QAction *viewExperimentAction;
    QAction *actionQuick_Experiment;
    QAction *actionStart_Sequence;
    QAction *actionFtmwConfig;
    QToolButton *hardwareButton;
    QToolButton *viewExperimentButton;
    QToolButton *settingsButton;
    QAction *appConfigAction;
    QWidget *centralWidget;
    QHBoxLayout *mainLayout;
    QVBoxLayout *instrumentStatusLayout;
    QLabel *instStatusLabel;
    QGridLayout *statusLayout;
    QLabel *exptLabel;
    QLabel *exptValueLabel;
    QToolButton *exptConfigButton;
    QLabel *dataLabel;
    QLabel *savePathLabel;
    ClockDisplayBox *clockBox;
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
    QMenu *menuLoadout;
    QMenu *menuAcquisition;
    QMenu *menuRollingData;
    QMenu *menuAuxData;
    QMenu *settingsMenu;
    QMenu *viewExperimentMenu;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;
    QScrollArea *hwStatusScrollArea;
    QWidget *hwStatusWidget;
    QVBoxLayout *hwStatusLayout;


    QWidget *lifTab{nullptr};
    QAction *actionLifConfig{nullptr};
    QAction *actionRuntimeHardwareConfig;
    QProgressBar *lifProgressBar{nullptr};
    QLayout *lifTabLayout{nullptr};
    LifDisplayWidget *lifDisplayWidget{nullptr};
    QLabel *lifProgressLabel{nullptr};

    void setupUi(QMainWindow *MainWindow)
    {
        using namespace Qt::Literals::StringLiterals;
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        // BlackChirp branding icon set programmatically in setupThemeAwareIconStyling()
        MainWindow->setWindowIcon(QIcon());
        actionStart_Experiment = new QAction(MainWindow);
        actionStart_Experiment->setObjectName(QString::fromUtf8("actionStart_Experiment"));
        // Icon set programmatically in setupThemeAwareIconStyling()
        pauseButton = new QToolButton(MainWindow);
        pauseButton->setObjectName(QString::fromUtf8("actionPause"));
        pauseButton->setToolTip("Pause acquisition");
        // Icon set programmatically in setupThemeAwareIconStyling()
        resumeButton = new QToolButton(MainWindow);
        resumeButton->setObjectName(QString::fromUtf8("actionResume"));
        resumeButton->setToolTip("Resume acquisition");
        // Icon set programmatically in setupThemeAwareIconStyling()
        abortButton = new QToolButton(MainWindow);
        abortButton->setObjectName(QString::fromUtf8("actionAbort"));
        abortButton->setToolTip("Abort acquistiion");
        // Icon set programmatically in setupThemeAwareIconStyling()
        actionCommunication = new QAction(MainWindow);
        actionCommunication->setObjectName(QString::fromUtf8("actionCommunication"));
        // Icon set programmatically in setupThemeAwareIconStyling()
        // Icon set programmatically in setupThemeAwareIconStyling()
        QIcon icon7;
        icon7.addFile(QString::fromUtf8(":/icons/num.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionTest_All_Connections = new QAction(MainWindow);
        actionTest_All_Connections->setObjectName(QString::fromUtf8("actionTest_All_Connections"));
        // Icon set programmatically in setupThemeAwareIconStyling()
        sleepButton = new QToolButton(MainWindow);
        sleepButton->setObjectName(QString::fromUtf8("actionSleep"));
        sleepButton->setCheckable(true);
        // Icon set programmatically in setupThemeAwareIconStyling()
        sleepButton->setToolTip("Toggle sleep mode, putting hardware in a standby state.\nIf pressed during an acquisition, sleep mode will be activated when the acquisition is complete.");
        actionAutoscale_Aux = new QAction(MainWindow);
        actionAutoscale_Aux->setObjectName(QString::fromUtf8("actionAutoscale_Aux"));
        QIcon icon10;
        icon10.addFile(QString::fromUtf8(":/icons/autoscale.png"), QSize(), QIcon::Normal, QIcon::Off);
        actionAutoscale_Aux->setIcon(icon10);
        actionAutoscale_Rolling = new QAction(MainWindow);
        actionAutoscale_Rolling->setObjectName(QString::fromUtf8("actionAutoscale_Rolling"));
        actionAutoscale_Rolling->setIcon(icon10);
        actionFtmwConfig = new QAction("FTMW Configuration",MainWindow);
        actionFtmwConfig->setObjectName("ActionFtmwConfig");
        // Icon set programmatically in setupThemeAwareIconStyling()
        // Icons set programmatically in setupThemeAwareIconStyling()
        actionQuick_Experiment = new QAction(MainWindow);
        actionQuick_Experiment->setObjectName(QString::fromUtf8("actionQuick_Experiment"));
        // Icon set programmatically in setupThemeAwareIconStyling()
        actionStart_Sequence = new QAction(MainWindow);
        actionStart_Sequence->setObjectName(QString::fromUtf8("actionStart_Sequence"));
        // Icon set programmatically in setupThemeAwareIconStyling()
        if(ApplicationConfigManager::instance().isLifEnabled())
        {
            actionLifConfig = new QAction("LIF Configuration",MainWindow);
            // Icon set programmatically in setupThemeAwareIconStyling()
            actionLifConfig->setObjectName("ActionLifConfig");
            // Icon set programmatically in setupThemeAwareIconStyling()
        }

        actionRuntimeHardwareConfig = new QAction("Hardware Selection",MainWindow);
        actionRuntimeHardwareConfig->setObjectName("ActionRuntimeHardwareConfig");
        // Icon set programmatically in setupThemeAwareIconStyling()

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
        // Icon set programmatically in setupThemeAwareIconStyling()
        acquireButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        acquireButton->setPopupMode(QToolButton::InstantPopup);

        hardwareButton = new QToolButton(MainWindow);
        hardwareButton->setText("Hardware");
        hardwareButton->setToolTip("Configure hardware settings");
        // Icon set programmatically in setupThemeAwareIconStyling()
        hardwareButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        
        viewExperimentButton = new QToolButton(MainWindow);
        viewExperimentButton->setText("View Experiment");
        viewExperimentButton->setToolTip("View existing experiments");
        // Icon set programmatically in setupThemeAwareIconStyling()
        viewExperimentButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        viewExperimentButton->setPopupMode(QToolButton::InstantPopup);
        hardwareButton->setPopupMode(QToolButton::InstantPopup);

        instStatusLabel->setAlignment(Qt::AlignCenter);

        auxPlotButton = new QToolButton(MainWindow);
        auxPlotButton->setText(QString("Aux Data"));
        auxPlotButton->setToolTip("Configure settings for aux data plots");
        // Icon set programmatically in setupThemeAwareIconStyling()
        auxPlotButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        auxPlotButton->setPopupMode(QToolButton::InstantPopup);

        rollingPlotButton = new QToolButton(MainWindow);
        rollingPlotButton->setText(QString("Rolling Data"));
        rollingPlotButton->setToolTip("Configure settings for rolling data plots");
        // Icon set programmatically in setupThemeAwareIconStyling()
        rollingPlotButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        rollingPlotButton->setPopupMode(QToolButton::InstantPopup);

        settingsButton = new QToolButton(MainWindow);
        // Icon set programmatically in setupThemeAwareIconStyling()
        settingsButton->setText("Settings");
        settingsButton->setToolTip("Configure application and miscellaneous settings");
        settingsButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        settingsButton->setPopupMode(QToolButton::InstantPopup);

        appConfigAction = new QAction("Application Settings");
        // Icon set programmatically in setupThemeAwareIconStyling()


        instrumentStatusLayout->addWidget(instStatusLabel);

        hwStatusScrollArea = new QScrollArea(centralWidget);
        hwStatusScrollArea->setObjectName("HwStatusScrollArea");
        hwStatusScrollArea->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
        hwStatusScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        hwStatusScrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
        hwStatusScrollArea->setWidgetResizable(true);

        hwStatusWidget = new QWidget;

        hwStatusLayout = new QVBoxLayout;
        hwStatusWidget->setLayout(hwStatusLayout);
        hwStatusScrollArea->setWidget(hwStatusWidget);

        clockBox = new ClockDisplayBox;
        hwStatusLayout->addWidget(clockBox,0);

        instrumentStatusLayout->addWidget(hwStatusScrollArea,0);

        statusLayout = new QGridLayout();
        statusLayout->setSpacing(6);
        statusLayout->setVerticalSpacing(2);
        statusLayout->setObjectName(QString::fromUtf8("statusLayout"));

        exptLabel = new QLabel(centralWidget);
        exptLabel->setObjectName(QString::fromUtf8("exptLabel"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(exptLabel->sizePolicy().hasHeightForWidth());
        exptLabel->setSizePolicy(sizePolicy);
        exptLabel->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        {
            auto f = exptLabel->font();
            f.setWeight(QFont::Bold);
            exptLabel->setFont(f);
        }

        statusLayout->addWidget(exptLabel,0,0);

        exptValueLabel = new QLabel(centralWidget);
        exptValueLabel->setObjectName(QString::fromUtf8("exptValueLabel"));
        QSizePolicy sizePolicy1(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(exptValueLabel->sizePolicy().hasHeightForWidth());
        exptValueLabel->setSizePolicy(sizePolicy1);
        exptValueLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
        exptValueLabel->setAlignment(Qt::AlignLeft|Qt::AlignVCenter);
        {
            auto f = exptValueLabel->font();
            f.setWeight(QFont::Bold);
            exptValueLabel->setFont(f);
        }

        statusLayout->addWidget(exptValueLabel,0,1);

        exptConfigButton = new QToolButton(centralWidget);
        exptConfigButton->setObjectName(QString::fromUtf8("exptConfigButton"));
        exptConfigButton->setAutoRaise(true);
        exptConfigButton->setIconSize(QSize(16,16));
        exptConfigButton->setToolTip(QApplication::translate("MainWindow", "Open application configuration", nullptr));

        statusLayout->addWidget(exptConfigButton,0,2);

        dataLabel = new QLabel("Data"_L1, centralWidget);
        dataLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        {
            auto f = dataLabel->font();
            f.setWeight(QFont::Bold);
            dataLabel->setFont(f);
        }

        statusLayout->addWidget(dataLabel,1,0);

        savePathLabel = new QLabel(centralWidget);
        savePathLabel->setObjectName(QString::fromUtf8("savePathLabel"));
        savePathLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);

        statusLayout->addWidget(savePathLabel,1,1,1,2);

        instrumentStatusLayout->addLayout(statusLayout);

        {
            auto *separator = new QFrame(centralWidget);
            separator->setFrameShape(QFrame::HLine);
            separator->setFrameShadow(QFrame::Plain);
            separator->setLineWidth(1);
            separator->setStyleSheet(QString("QFrame { color: %1; }").arg(ThemeColors::getCSSColor(ThemeColors::SubtleText, centralWidget)));
            instrumentStatusLayout->addWidget(separator);
        }

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
        ftViewWidget = new FtmwViewWidget(true,ftmwTab);
        ftViewWidget->setObjectName(QString::fromUtf8("ftViewWidget"));

        ftmwTabLayout->addWidget(ftViewWidget);

        mainTabWidget->addTab(ftmwTab, QIcon(), QString()); // Icon set programmatically

        if(ApplicationConfigManager::instance().isLifEnabled())
        {
            lifTab = new QWidget();
            lifTab->setObjectName(QString("lifTab"));
            lifTabLayout = new QVBoxLayout(lifTab);
            lifDisplayWidget = new LifDisplayWidget();
            lifDisplayWidget->setObjectName(QString("lifDisplayWidget"));
            lifTabLayout->addWidget(lifDisplayWidget);
            mainTabWidget->addTab(lifTab, QIcon(), QString()); // Icon set programmatically
            mainTabWidget->setTabText(mainTabWidget->indexOf(lifTab), QString("LIF"));

            lifProgressLabel = new QLabel(QString("LIF Progress"));
            lifProgressBar = new QProgressBar();
            lifProgressBar->setRange(0, 1000);
            lifProgressBar->setValue(0);
            instrumentStatusLayout->addWidget(lifProgressLabel, 0, Qt::AlignCenter);
            instrumentStatusLayout->addWidget(lifProgressBar);
        }

        rollingDataTab = new QWidget();
        rollingDataTab->setObjectName(QString::fromUtf8("rollingDataTab"));
        rollingDataViewLayout = new QVBoxLayout(rollingDataTab);
        rollingDataViewLayout->setSpacing(6);
        rollingDataViewLayout->setContentsMargins(11, 11, 11, 11);
        rollingDataViewLayout->setObjectName(QString::fromUtf8("rollingDataViewLayout"));
        rollingDataViewWidget = new RollingDataWidget(BC::Key::rollingDataWidget,rollingDataTab);
        rollingDataViewWidget->setObjectName(QString::fromUtf8("rollingDataViewWidget"));

        rollingDataViewLayout->addWidget(rollingDataViewWidget);

        mainTabWidget->addTab(rollingDataTab, QIcon(), QString()); // Icon set programmatically


        auxDataTab = new QWidget();
        auxDataTab->setObjectName(QString::fromUtf8("auxDataTab"));
        auxDataViewLayout = new QVBoxLayout(auxDataTab);
        auxDataViewLayout->setSpacing(6);
        auxDataViewLayout->setContentsMargins(11, 11, 11, 11);
        auxDataViewLayout->setObjectName(QString::fromUtf8("auxDataViewLayout"));
        auxDataViewWidget = new AuxDataViewWidget(BC::Key::auxDataWidget,auxDataTab);
        auxDataViewWidget->setObjectName(QString::fromUtf8("auxDataViewWidget"));

        auxDataViewLayout->addWidget(auxDataViewWidget);

        mainTabWidget->addTab(auxDataTab, QIcon(), QString()); // Icon set programmatically



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
        menuLoadout = new QMenu("Loadout"_L1, menuHardware);
        menuAcquisition = new QMenu(acquireButton);
        menuAcquisition->setObjectName(QString::fromUtf8("menuAcquisition"));

        menuAuxData = new QMenu(auxPlotButton);
        menuAuxData->setObjectName(QString::fromUtf8("menuAuxData"));
        // Icon set programmatically in setupThemeAwareIconStyling()
        menuRollingData = new QMenu(rollingPlotButton);
        menuRollingData->setObjectName(QString::fromUtf8("menuRollingData"));
        // Icon set programmatically in setupThemeAwareIconStyling()
//        MainWindow->setMenuBar(menuBar);

        settingsMenu = new QMenu(settingsButton);
        settingsMenu->addAction(appConfigAction);
        
        viewExperimentMenu = new QMenu(MainWindow);
        viewExperimentMenu->setObjectName(QString::fromUtf8("viewExperimentMenu"));
        
        // Add the permanent "View Experiment..." action
        viewExperimentAction = new QAction("View Experiment...", MainWindow);
        // Icon set programmatically in setupThemeAwareIconStyling()
        viewExperimentMenu->addAction(viewExperimentAction);
        viewExperimentMenu->addSeparator();

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

        menuHardware->addAction(actionRuntimeHardwareConfig);
        menuHardware->addMenu(menuLoadout);
        menuHardware->addSeparator();
        menuHardware->addAction(actionCommunication);
        menuHardware->addAction(actionTest_All_Connections);
        menuHardware->addSeparator();
        menuHardware->addAction(actionFtmwConfig);
        if(actionLifConfig)
            menuHardware->addAction(actionLifConfig);
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

        mainToolBar->addWidget(viewExperimentButton);
        viewExperimentButton->setMenu(viewExperimentMenu);

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
            mainTabWidget->widget(i)->layout()->setContentsMargins(0,0,0,0);


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
        actionQuick_Experiment->setText(QApplication::translate("MainWindow", "&Quick Experiment", nullptr));
#ifndef QT_NO_SHORTCUT
        actionQuick_Experiment->setShortcut(QApplication::translate("MainWindow", "F3", nullptr));
#endif // QT_NO_SHORTCUT
        actionStart_Sequence->setText(QApplication::translate("MainWindow", "Start Seq&uence", nullptr));
        instStatusLabel->setText(QApplication::translate("MainWindow", "Instrument Status", nullptr));
        exptLabel->setText(QApplication::translate("MainWindow", "Experiment", nullptr));
#ifndef QT_NO_TOOLTIP
        exptValueLabel->setToolTip(QApplication::translate("MainWindow", "Number of the most recent experiment", nullptr));
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
