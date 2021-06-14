#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QThread>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QCloseEvent>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QDialog>
#include <QMessageBox>
#include <QCheckBox>
#include <QToolButton>
#include <QFileDialog>
#include <QDir>

#include <src/gui/dialog/communicationdialog.h>
#include <src/gui/dialog/ioboardconfigdialog.h>
#include <src/gui/widget/digitizerconfigwidget.h>
#include <src/gui/widget/rfconfigwidget.h>
#include <src/gui/wizard/experimentwizard.h>
#include <src/data/loghandler.h>
#include <src/hardware/core/hardwaremanager.h>
#include <src/acquisition/acquisitionmanager.h>
#include <src/acquisition/batch/batchmanager.h>
#include <src/acquisition/batch/batchsingle.h>
#include <src/acquisition/batch/batchsequence.h>
#include <src/gui/widget/led.h>
#include <src/gui/widget/experimentviewwidget.h>
#include <src/gui/dialog/quickexptdialog.h>
#include <src/gui/dialog/batchsequencedialog.h>
#include <src/gui/widget/clockdisplaywidget.h>

#ifdef BC_LIF
#include <src/modules/lif/gui/lifdisplaywidget.h>
#include <src/modules/lif/gui/lifcontrolwidget.h>
#endif

#ifdef BC_MOTOR
#include <src/modules/motor/gui/motordisplaywidget.h>
#include <src/modules/motor/gui/motorstatuswidget.h>
#endif

#ifdef BC_PCONTROLLER
#include <src/hardware/optional/pressurecontroller/pressurecontroller.h>
#endif

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow), d_hardwareConnected(false), d_oneExptDone(false), d_state(Idle), d_logCount(0), d_logIcon(BlackChirp::LogNormal), d_currentExptNum(0)
{
    p_pcBox = nullptr;
    p_hwm = new HardwareManager();

    ui->setupUi(this);

    for(int i=0; i<ui->tabWidget->count(); i++)
    {
        ui->tabWidget->widget(i)->layout()->setContentsMargins(0,0,0,0);
        ui->tabWidget->widget(i)->layout()->setMargin(0);
    }

    ui->exptSpinBox->blockSignals(true);

    auto vc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);

    QLabel *statusLabel = new QLabel(this);
    connect(this,&MainWindow::statusMessage,statusLabel,&QLabel::setText);
    ui->statusBar->addWidget(statusLabel);

    p_lh = new LogHandler();
    connect(p_lh,&LogHandler::sendLogMessage,ui->log,&QTextEdit::append);
    connect(p_lh,&LogHandler::iconUpdate,this,&MainWindow::setLogIcon);
    connect(ui->ftViewWidget,&FtmwViewWidget::experimentLogMessage,p_lh,&LogHandler::experimentLogMessage);
    connect(ui->tabWidget,&QTabWidget::currentChanged,[=](int i) {
        if(i == ui->tabWidget->indexOf(ui->logTab))
        {
            setLogIcon(BlackChirp::LogNormal);
            if(d_logCount > 0)
            {
                d_logCount = 0;
                ui->tabWidget->setTabText(ui->tabWidget->indexOf(ui->logTab),QString("Log"));
            }
        }
    });
    connect(p_lh,&LogHandler::sendLogMessage,this,[=](){
        if(ui->tabWidget->currentWidget() != ui->logTab)
        {
            d_logCount++;
            ui->tabWidget->setTabText(ui->tabWidget->indexOf(ui->logTab),QString("Log (%1)").arg(d_logCount));
        }
    });

    QGridLayout *gl = new QGridLayout;

    SettingsStorage pg(BC::Key::pGen,SettingsStorage::Hardware);
    for(int i=0; i<pg.get<int>(BC::Key::pGenChannels,8); i++)
    {
        QLabel *lbl = new QLabel(QString("Ch%1").arg(i),this);
        lbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);

        Led *led = new Led(this);
        gl->addWidget(lbl,i/4,(2*i)%8,1,1,Qt::AlignVCenter);
        gl->addWidget(led,i/4,((2*i)%8)+1,1,1,Qt::AlignVCenter);

        d_ledList.append(qMakePair(lbl,led));
    }
    for(int i=0; i<8; i++)
        gl->setColumnStretch(i,(i%2)+1);

    gl->setMargin(3);
    gl->setContentsMargins(3,3,3,3);
    gl->setSpacing(3);
    ui->pulseConfigBox->setLayout(gl);

    auto *clockBox = new QGroupBox(QString("Clocks"),this);
    auto *clockWidget = new ClockDisplayWidget(this);
    clockBox->setLayout(clockWidget->layout());
    ui->instrumentStatusLayout->insertWidget(2,clockBox);


    QThread *lhThread = new QThread(this);
    connect(lhThread,&QThread::finished,p_lh,&LogHandler::deleteLater);
    p_lh->moveToThread(lhThread);
    d_threadObjectList.append(qMakePair(lhThread,p_lh));
    lhThread->start();

    connect(p_hwm,&HardwareManager::logMessage,p_lh,&LogHandler::logMessage);
    connect(p_hwm,&HardwareManager::statusMessage,statusLabel,&QLabel::setText);
    connect(p_hwm,&HardwareManager::hwInitializationComplete,ui->pulseConfigWidget,&PulseConfigWidget::updateHardwareLimits);
    connect(p_hwm,&HardwareManager::allHardwareConnected,this,&MainWindow::hardwareInitialized);
    connect(p_hwm,&HardwareManager::clockFrequencyUpdate,clockWidget,&ClockDisplayWidget::updateFrequency);
    connect(p_hwm,&HardwareManager::pGenConfigUpdate,ui->pulseConfigWidget,&PulseConfigWidget::setFromConfig);
    connect(p_hwm,&HardwareManager::pGenSettingUpdate,ui->pulseConfigWidget,&PulseConfigWidget::newSetting);
    connect(p_hwm,&HardwareManager::pGenConfigUpdate,this,&MainWindow::updatePulseLeds);
    connect(p_hwm,&HardwareManager::pGenSettingUpdate,this,&MainWindow::updatePulseLed);
    connect(p_hwm,&HardwareManager::flowUpdate,this,&MainWindow::updateFlow);
    connect(p_hwm,&HardwareManager::flowNameUpdate,this,&MainWindow::updateFlowName);
    connect(p_hwm,&HardwareManager::flowSetpointUpdate,this,&MainWindow::updateFlowSetpoint);
    connect(p_hwm,&HardwareManager::gasPressureUpdate,ui->pressureDoubleSpinBox,&QDoubleSpinBox::setValue);
    connect(p_hwm,&HardwareManager::gasPressureSetpointUpdate,this,&MainWindow::updatePressureSetpoint);
    connect(p_hwm,&HardwareManager::gasPressureControlMode,this,&MainWindow::updatePressureControl);
    connect(ui->pressureControlButton,&QPushButton::clicked,p_hwm,&HardwareManager::setGasPressureControlMode);
    connect(ui->pressureControlBox,vc,p_hwm,&HardwareManager::setGasPressureSetpoint);
    connect(p_hwm,&HardwareManager::pGenRepRateUpdate,ui->pulseConfigWidget,&PulseConfigWidget::newRepRate);
    connect(ui->pulseConfigWidget,&PulseConfigWidget::changeSetting,p_hwm,&HardwareManager::setPGenSetting);
    connect(ui->pulseConfigWidget,&PulseConfigWidget::changeRepRate,p_hwm,&HardwareManager::setPGenRepRate);

#ifdef BC_PCONTROLLER
    SettingsStorage pc(BC::Key::pController,SettingsStorage::Hardware);
    configPController(pc.get<bool>(BC::Key::pControllerReadOnly,true));
#endif

    QThread *hwmThread = new QThread(this);
    connect(hwmThread,&QThread::started,p_hwm,&HardwareManager::initialize);
    connect(hwmThread,&QThread::finished,p_hwm,&HardwareManager::deleteLater);
    p_hwm->moveToThread(hwmThread);
    d_threadObjectList.append(qMakePair(hwmThread,p_hwm));

    gl = static_cast<QGridLayout*>(ui->gasControlBox->layout());
    QGridLayout *gl2 = static_cast<QGridLayout*>(ui->flowStatusBox->layout());
    gl2->setMargin(3);
    gl2->setSpacing(3);
    gl2->setContentsMargins(3,3,3,3);
    gl->setContentsMargins(3,3,3,3);
    gl->setMargin(3);
    gl->setSpacing(3);
    QWidget *lastFocusWidget = nullptr;
    SettingsStorage fc(BC::Key::flowController,SettingsStorage::Hardware);
    int flowChannels = fc.get<int>(BC::Key::flowChannels,4);
    for(int i=0; i<flowChannels; i++)
    {
        FlowWidgets fw;
        fw.nameEdit = new QLineEdit(this);
        connect(fw.nameEdit,&QLineEdit::editingFinished,[=](){
            QMetaObject::invokeMethod(p_hwm,"setFlowChannelName",Q_ARG(int,i),Q_ARG(QString,fw.nameEdit->text()));
        });

        fw.controlBox = new QDoubleSpinBox(this);
        fw.controlBox->setSuffix(QString(" sccm"));
        fw.controlBox->setRange(0.0,10000.0);
        fw.controlBox->setSpecialValueText(QString("Off"));
        fw.controlBox->setKeyboardTracking(false);
        connect(fw.controlBox,vc,[=](double val){
            QMetaObject::invokeMethod(p_hwm,"setFlowSetpoint",Q_ARG(int,i),Q_ARG(double,val));
        });
        lastFocusWidget = fw.controlBox;

        fw.nameLabel = new QLabel(QString("Ch%1").arg(i+1),this);
        fw.nameLabel->setMinimumWidth(QFontMetrics(QFont(QString("sans-serif"))).width(QString("MMMMMMMM")));
        fw.nameLabel->setAlignment(Qt::AlignRight|Qt::AlignVCenter);

        fw.led = new Led(this);

        fw.displayBox = new QDoubleSpinBox(this);
        fw.displayBox->setRange(-9999.9,9999.9);
        fw.displayBox->setDecimals(2);
        fw.displayBox->setSuffix(QString(" sccm"));
        fw.displayBox->blockSignals(true);
        fw.displayBox->setReadOnly(true);
        fw.displayBox->setFocusPolicy(Qt::ClickFocus);
        fw.displayBox->setButtonSymbols(QAbstractSpinBox::NoButtons);

        d_flowWidgets.append(fw);

        gl->addWidget(new QLabel(QString::number(i+1),this),1+i,0,1,1);
        gl->addWidget(fw.nameEdit,i+1,1,1,1);
        gl->addWidget(fw.controlBox,i+1,2,1,1);

        gl2->addWidget(fw.nameLabel,i+1,0,1,1,Qt::AlignRight);
        gl2->addWidget(fw.displayBox,i+1,1,1,1);
        gl2->addWidget(fw.led,i+1,2,1,1);
    }
    gl->addWidget(new QLabel(QString("Pressure"),this),2+flowChannels,1,1,1,Qt::AlignRight);
    gl->addWidget(ui->pressureControlBox,2+flowChannels,2,1,1);
    gl->addWidget(new QLabel(QString("Pressure Control Mode"),this),3+flowChannels,1,1,1,Qt::AlignRight);
    gl->addWidget(ui->pressureControlButton,3+flowChannels,2,1,1);
    gl->addItem(new QSpacerItem(10,10,QSizePolicy::Minimum,QSizePolicy::Expanding),4+flowChannels,0,1,3);
    if(lastFocusWidget != nullptr)
        setTabOrder(lastFocusWidget,ui->pressureControlBox);

    setTabOrder(ui->pressureControlBox,ui->pressureControlButton);
    setTabOrder(ui->pressureControlButton,ui->pulseConfigWidget);

    ui->pressureDoubleSpinBox->blockSignals(true);
    connect(ui->pressureControlButton,&QPushButton::toggled,[=](bool en){
        if(en)
            ui->pressureControlButton->setText(QString("On"));
        else
            ui->pressureControlButton->setText(QString("Off"));
    });

    p_am = new AcquisitionManager();
    connect(p_am,&AcquisitionManager::logMessage,p_lh,&LogHandler::logMessage);
    connect(p_am,&AcquisitionManager::statusMessage,statusLabel,&QLabel::setText);
    connect(p_am,&AcquisitionManager::experimentInitialized,this,&MainWindow::experimentInitialized);
    connect(p_am,&AcquisitionManager::ftmwUpdateProgress,ui->ftmwProgressBar,&QProgressBar::setValue);
//    connect(p_am,&AcquisitionManager::ftmwNumShots,ui->ftViewWidget,&FtmwViewWidget::updateShotsLabel);
    connect(ui->actionPause,&QAction::triggered,p_am,&AcquisitionManager::pause);
    connect(ui->actionResume,&QAction::triggered,p_am,&AcquisitionManager::resume);
    connect(ui->actionAbort,&QAction::triggered,p_am,&AcquisitionManager::abort);
    connect(ui->ftViewWidget,&FtmwViewWidget::rollingAverageShotsChanged,p_am,&AcquisitionManager::changeRollingAverageShots);
    connect(ui->ftViewWidget,&FtmwViewWidget::rollingAverageReset,p_am,&AcquisitionManager::resetRollingAverage);
    connect(p_am,&AcquisitionManager::newFtmwConfig,ui->ftViewWidget,&FtmwViewWidget::updateFtmw);
    connect(p_am,&AcquisitionManager::newFidList,ui->ftViewWidget,&FtmwViewWidget::updateLiveFidList);
    connect(p_am,&AcquisitionManager::snapshotComplete,ui->ftViewWidget,&FtmwViewWidget::snapshotTaken);
    connect(p_am,&AcquisitionManager::doFinalSave,ui->ftViewWidget,&FtmwViewWidget::experimentComplete);

    QThread *amThread = new QThread(this);
    connect(amThread,&QThread::finished,p_am,&AcquisitionManager::deleteLater);
    p_am->moveToThread(amThread);
    d_threadObjectList.append(qMakePair(amThread,p_am));


    connect(p_hwm,&HardwareManager::experimentInitialized,p_am,&AcquisitionManager::beginExperiment);
    connect(p_hwm,&HardwareManager::ftmwScopeShotAcquired,p_am,&AcquisitionManager::processFtmwScopeShot);
    connect(p_am,&AcquisitionManager::newClockSettings,p_hwm,&HardwareManager::setClocks);
    connect(p_hwm,&HardwareManager::allClocksReady,p_am,&AcquisitionManager::clockSettingsComplete);
    connect(p_am,&AcquisitionManager::beginAcquisition,p_hwm,&HardwareManager::beginAcquisition);
    connect(p_am,&AcquisitionManager::endAcquisition,p_hwm,&HardwareManager::endAcquisition);
    connect(p_am,&AcquisitionManager::timeDataSignal,p_hwm,&HardwareManager::getTimeData);
    connect(p_hwm,&HardwareManager::timeData,p_am,&AcquisitionManager::processTimeData);


    connect(this,&MainWindow::startInit,[=](){
        hwmThread->start();
        amThread->start();
    });

    d_batchThread = new QThread(this);

    connect(ui->actionStart_Experiment,&QAction::triggered,this,&MainWindow::startExperiment);
    connect(ui->actionQuick_Experiment,&QAction::triggered,this,&MainWindow::quickStart);
    connect(ui->actionStart_Sequence,&QAction::triggered,this,&MainWindow::startSequence);
    connect(ui->actionPause,&QAction::triggered,this,&MainWindow::pauseUi);
    connect(ui->actionResume,&QAction::triggered,this,&MainWindow::resumeUi);
    connect(ui->actionCommunication,&QAction::triggered,this,&MainWindow::launchCommunicationDialog);
    connect(ui->actionCP_FTMW,&QAction::triggered,this,[=](){ ui->tabWidget->setCurrentWidget(ui->ftmwTab); });
    connect(ui->actionTrackingShow,&QAction::triggered,this,[=](){ ui->tabWidget->setCurrentWidget(ui->trackingTab); });
    connect(ui->actionControl,&QAction::triggered,this,[=](){ ui->tabWidget->setCurrentWidget(ui->controlTab); });
    connect(ui->actionLog,&QAction::triggered,this,[=](){ ui->tabWidget->setCurrentWidget(ui->logTab); });
    connect(ui->action_Graphs,&QAction::triggered,ui->trackingViewWidget,&TrackingViewWidget::changeNumPlots);
    connect(ui->actionAutoscale_All,&QAction::triggered,ui->trackingViewWidget,&TrackingViewWidget::autoScaleAll);
    connect(ui->actionSleep,&QAction::toggled,this,&MainWindow::sleep);
    connect(ui->actionTest_All_Connections,&QAction::triggered,p_hwm,&HardwareManager::testAll);
    connect(ui->actionView_Experiment,&QAction::triggered,this,&MainWindow::viewExperiment);
#ifdef BC_LIF
    p_lifDisplayWidget = new LifDisplayWidget(this);
    int lti = ui->tabWidget->insertTab(ui->tabWidget->indexOf(ui->trackingTab),p_lifDisplayWidget,QIcon(QString(":/icons/laser.png")),QString("LIF"));
    p_lifTab = ui->tabWidget->widget(lti);
    p_lifProgressBar = new QProgressBar(this);
    ui->instrumentStatusLayout->addWidget(new QLabel(QString("LIF Progress")),0,Qt::AlignCenter);
    ui->instrumentStatusLayout->addWidget(p_lifProgressBar);
    p_lifControlWidget = new LifControlWidget(this);
    ui->controlTopLayout->addWidget(p_lifControlWidget,2);
    p_lifAction = new QAction(QIcon(QString(":/icons/laser.png")),QString("LIF"),this);
    ui->menuView->insertAction(ui->actionLog,p_lifAction);

    connect(p_hwm,&HardwareManager::hwInitializationComplete,p_lifControlWidget,&LifControlWidget::updateHardwareLimits);
    connect(p_hwm,&HardwareManager::lifScopeShotAcquired,p_lifControlWidget,&LifControlWidget::newTrace);
    connect(p_hwm,&HardwareManager::lifScopeConfigUpdated,p_lifControlWidget,&LifControlWidget::scopeConfigChanged);
    connect(p_hwm,&HardwareManager::lifLaserPosUpdate,p_lifControlWidget,&LifControlWidget::setLaserPos);
    connect(p_lifControlWidget,&LifControlWidget::updateScope,p_hwm,&HardwareManager::setLifScopeConfig);
    connect(p_lifControlWidget,&LifControlWidget::laserPosUpdate,p_hwm,&HardwareManager::setLifLaserPos);
    connect(p_hwm,&HardwareManager::lifSettingsComplete,p_lifDisplayWidget,&LifDisplayWidget::resetLifPlot);
    connect(p_hwm,&HardwareManager::lifSettingsComplete,p_am,&AcquisitionManager::lifHardwareReady);
    connect(p_hwm,&HardwareManager::lifScopeShotAcquired,p_am,&AcquisitionManager::processLifScopeShot);
    connect(p_am,&AcquisitionManager::nextLifPoint,p_hwm,&HardwareManager::setLifParameters);
    connect(p_am,&AcquisitionManager::lifShotAcquired,p_lifProgressBar,&QProgressBar::setValue);
    connect(p_am,&AcquisitionManager::lifPointUpdate,p_lifDisplayWidget,&LifDisplayWidget::updatePoint);
    connect(p_lifAction,&QAction::triggered,this,[=](){ ui->tabWidget->setCurrentWidget(p_lifTab); });
    connect(p_lifControlWidget,&LifControlWidget::lifColorChanged,
            p_lifDisplayWidget,&LifDisplayWidget::checkLifColors);
    connect(p_lifDisplayWidget,&LifDisplayWidget::lifColorChanged,
            p_lifControlWidget,&LifControlWidget::checkLifColors);
#else
    ui->controlTopLayout->addStretch(1);
#endif

#ifdef BC_MOTOR
    p_motorDisplayWidget = new MotorDisplayWidget(this);
    int mti = ui->tabWidget->insertTab(ui->tabWidget->indexOf(ui->trackingTab),p_motorDisplayWidget,QIcon(QString(":/icons/motorscan.png")),QString("Motor"));
    p_motorTab = ui->tabWidget->widget(mti);

    p_motorViewAction = new QAction(QIcon(QString(":/icons/motorscan.png")),QString("Motor"),this);
    ui->menuView->insertAction(ui->actionLog,p_motorViewAction);
    connect(p_motorViewAction,&QAction::triggered,[=](){ ui->tabWidget->setCurrentWidget(p_motorTab);});

    p_motorStatusWidget = new MotorStatusWidget(this);
    ui->instrumentStatusLayout->addWidget(p_motorStatusWidget);

    connect(p_hwm,&HardwareManager::motorMoveComplete,p_am,&AcquisitionManager::motorMoveComplete);
    connect(p_hwm,&HardwareManager::motorTraceAcquired,p_am,&AcquisitionManager::motorTraceReceived);
    connect(p_am,&AcquisitionManager::startMotorMove,p_hwm,&HardwareManager::moveMotorToPosition);
    connect(p_am,&AcquisitionManager::motorRest,p_hwm,&HardwareManager::motorRest);
    connect(p_hwm,&HardwareManager::motorPosUpdate,p_motorStatusWidget,&MotorStatusWidget::updatePosition);
    connect(p_hwm,&HardwareManager::motorLimitStatus,p_motorStatusWidget,&MotorStatusWidget::updateLimit);
    connect(p_am,&AcquisitionManager::motorProgress,p_motorStatusWidget,&MotorStatusWidget::updateProgress);
    connect(p_am,&AcquisitionManager::motorDataUpdate,p_motorDisplayWidget,&MotorDisplayWidget::newMotorData);
#endif

    SettingsStorage bc;
    ui->exptSpinBox->setValue(bc.get<int>(BC::Key::exptNum,0));
    configureUi(Idle);
}

MainWindow::~MainWindow()
{
    while(!d_threadObjectList.isEmpty())
    {
        QPair<QThread*,QObject*> p = d_threadObjectList.takeFirst();

        p.first->quit();
        p.first->wait();
    }

    delete ui;
}

void MainWindow::initializeHardware()
{
    emit statusMessage(QString("Initializing hardware..."));
    emit startInit();
}

void MainWindow::startExperiment()
{
    if(d_batchThread->isRunning())
        return;

    ExperimentWizard wiz(this);
    wiz.experiment.setPulseGenConfig(ui->pulseConfigWidget->getConfig());
    wiz.experiment.setFlowConfig(getFlowConfig());

#ifdef BC_LIF
    connect(p_hwm,&HardwareManager::lifScopeShotAcquired,&wiz,&ExperimentWizard::newTrace);
    connect(p_hwm,&HardwareManager::lifScopeConfigUpdated,&wiz,&ExperimentWizard::scopeConfigChanged);
    connect(p_hwm,&HardwareManager::lifLaserPosUpdate,&wiz,&ExperimentWizard::setCurrentLaserPos);
    connect(&wiz,&ExperimentWizard::updateScope,p_hwm,&HardwareManager::setLifScopeConfig);
    connect(&wiz,&ExperimentWizard::lifColorChanged,p_lifControlWidget,&LifControlWidget::checkLifColors);
    connect(&wiz,&ExperimentWizard::lifColorChanged,p_lifDisplayWidget,&LifDisplayWidget::checkLifColors);
    connect(&wiz,&ExperimentWizard::laserPosUpdate,p_hwm,&HardwareManager::setLifLaserPos);
    wiz.setCurrentLaserPos(p_lifControlWidget->laserPos());
#endif

    if(wiz.exec() != QDialog::Accepted)
        return;

    BatchManager *bm = new BatchSingle(wiz.experiment);
    startBatch(bm);
}

void MainWindow::quickStart()
{
    if(d_batchThread->isRunning())
        return;

    Experiment e = Experiment::loadFromSettings();
#ifdef BC_LIF
    if(e.lifConfig().isEnabled())
    {
        LifConfig lc = e.lifConfig();
        lc = p_lifControlWidget->getSettings(lc);
        e.setLifConfig(lc);
    }
#endif
    e.setFlowConfig(getFlowConfig());
    e.setPulseGenConfig(ui->pulseConfigWidget->getConfig());

    //create a popup summary of experiment.
    QuickExptDialog d(e,this);
    int ret = d.exec();

    if(ret == QDialog::Accepted)
    {
        BatchManager *bm = new BatchSingle(e);
        startBatch(bm);
    }
    else if(ret == d.configureResult())
        startExperiment();
}

void MainWindow::startSequence()
{
    if(d_batchThread->isRunning())
        return;

    BatchSequenceDialog d(this);
    d.setQuickExptEnabled(d_oneExptDone);
    int ret = d.exec();

    if(ret == QDialog::Rejected)
        return;

    Experiment exp;

    if(ret == d.quickCode)
    {
        Experiment e = Experiment::loadFromSettings();
#ifdef BC_LIF
        if(e.lifConfig().isEnabled())
        {
            LifConfig lc = e.lifConfig();
            lc = p_lifControlWidget->getSettings(lc);
            e.setLifConfig(lc);
        }
#endif
        e.setFlowConfig(getFlowConfig());
        e.setPulseGenConfig(ui->pulseConfigWidget->getConfig());

        //create a popup summary of experiment.
        QuickExptDialog qd(e,this);
        int qeret = qd.exec();

        if(qeret == QDialog::Accepted)
        {
            exp = e;
        }
        else if(qeret == qd.configureResult())
            ret = d.configureCode; //set ret to indicate that the experiment needs to be configured
        else if(qeret == QDialog::Rejected)
            return;
    }

    if(ret == d.configureCode)
    {
        ExperimentWizard wiz(this);
        wiz.experiment.setPulseGenConfig(ui->pulseConfigWidget->getConfig());
        wiz.experiment.setFlowConfig(getFlowConfig());
#ifdef BC_LIF
        connect(p_hwm,&HardwareManager::lifScopeShotAcquired,&wiz,&ExperimentWizard::newTrace);
        connect(p_hwm,&HardwareManager::lifScopeConfigUpdated,&wiz,&ExperimentWizard::scopeConfigChanged);
        connect(&wiz,&ExperimentWizard::updateScope,p_hwm,&HardwareManager::setLifScopeConfig);
        connect(&wiz,&ExperimentWizard::lifColorChanged,p_lifControlWidget,&LifControlWidget::checkLifColors);
        connect(&wiz,&ExperimentWizard::lifColorChanged,p_lifDisplayWidget,&LifDisplayWidget::checkLifColors);
#endif

        if(wiz.exec() != QDialog::Accepted)
            return;

        exp = wiz.experiment;
    }


    BatchSequence *bs = new BatchSequence();
    bs->setExperiment(exp);
    bs->setNumExperiments(d.get<int>(BC::Key::batchExperiments));
    bs->setInterval(d.get<int>(BC::Key::batchInterval));
    bs->setAutoExport(d.get<bool>(BC::Key::batchAutoExport));
    startBatch(bs);

}

void MainWindow::batchComplete(bool aborted)
{
    disconnect(p_hwm,&HardwareManager::timeData,ui->trackingViewWidget,&TrackingViewWidget::pointUpdated);
    disconnect(p_am,&AcquisitionManager::timeData,ui->trackingViewWidget,&TrackingViewWidget::pointUpdated);
    disconnect(p_hwm,&HardwareManager::abortAcquisition,p_am,&AcquisitionManager::abort);
    disconnect(ui->ftViewWidget,&FtmwViewWidget::rollingAverageShotsChanged,ui->ftmwProgressBar,&QProgressBar::setMaximum);

#ifdef BC_LIF
    p_lifTab->setEnabled(true);
#endif

    if(aborted)
        emit statusMessage(QString("Experiment aborted"));
    else
        emit statusMessage(QString("Experiment complete"));

    if(ui->ftmwProgressBar->maximum() == 0)
    {
	    ui->ftmwProgressBar->setRange(0,1);
	    ui->ftmwProgressBar->setValue(1);
    }

    ui->ftmwTab->setEnabled(true);

    d_oneExptDone = true;

    if(d_state == Acquiring)
        configureUi(Idle);
}

void MainWindow::experimentInitialized(const Experiment exp)
{   
    if(!exp.isInitialized())
		return;

    if(exp.number() > 0)
        ui->exptSpinBox->setValue(exp.number());

    d_currentExptNum = exp.number();

    ui->ftmwProgressBar->setValue(0);
    ui->ftViewWidget->prepareForExperiment(exp);

	if(exp.ftmwConfig().isEnabled())
	{
        switch(exp.ftmwConfig().type()) {
        case BlackChirp::FtmwTargetShots:
        case BlackChirp::FtmwLoScan:
        case BlackChirp::FtmwDrScan:
            ui->ftmwProgressBar->setRange(0,exp.ftmwConfig().targetShots());
            break;
        case BlackChirp::FtmwTargetTime:
            ui->ftmwProgressBar->setRange(0,static_cast<int>(exp.startTime().secsTo(exp.ftmwConfig().targetTime())));
            break;
        case BlackChirp::FtmwPeakUp:
            ui->ftmwProgressBar->setRange(0,exp.ftmwConfig().targetShots());
            connect(ui->ftViewWidget,&FtmwViewWidget::rollingAverageShotsChanged,ui->ftmwProgressBar,&QProgressBar::setMaximum,Qt::UniqueConnection);
            configureUi(Peaking);
            break;
        default:
			ui->ftmwProgressBar->setRange(0,0);
            break;
        }
	}
    else
    {
        ui->ftmwTab->setEnabled(false);
        ui->ftmwProgressBar->setRange(0,1);
        ui->ftmwProgressBar->setValue(1);
    }

#ifdef BC_LIF
    p_lifProgressBar->setValue(0);
    p_lifDisplayWidget->prepareForExperiment(exp.lifConfig());
    if(exp.lifConfig().isEnabled())
    {
        p_lifTab->setEnabled(true);
        p_lifProgressBar->setRange(0,exp.lifConfig().totalShots());
    }
    else
    {
        p_lifTab->setEnabled(false);
        p_lifProgressBar->setRange(0,1);
        p_lifProgressBar->setValue(1);
    }
#endif

#ifdef BC_MOTOR
    p_motorStatusWidget->prepareForExperiment(exp);
    p_motorDisplayWidget->prepareForScan(exp.motorScan());
    p_motorTab->setEnabled(exp.motorScan().isEnabled());
#endif

    if(exp.number() > 0)
    {
        if(p_lh->thread() == thread())
            p_lh->beginExperimentLog(exp);
        else
            QMetaObject::invokeMethod(p_lh,"beginExperimentLog",Q_ARG(const Experiment,exp));
    }
    else
    {
        if(p_lh->thread() == thread())
            p_lh->logMessage(exp.startLogMessage(),BlackChirp::LogHighlight);
        else
            QMetaObject::invokeMethod(p_lh,"logMessage",Q_ARG(const QString,exp.startLogMessage()),Q_ARG(BlackChirp::LogMessageCode,BlackChirp::LogHighlight));
    }
}

void MainWindow::hardwareInitialized(bool success)
{
	d_hardwareConnected = success;
    if(success)
        emit statusMessage(QString("Hardware connected"));
    else
        emit statusMessage(QString("Hardware error. See log for details."));
    configureUi(d_state);
}

void MainWindow::pauseUi()
{
    configureUi(Paused);
}

void MainWindow::resumeUi()
{
    configureUi(Acquiring);
}

void MainWindow::launchCommunicationDialog()
{
    CommunicationDialog d(this);
    connect(&d,&CommunicationDialog::testConnection,p_hwm,&HardwareManager::testObjectConnection);
    connect(p_hwm,&HardwareManager::testComplete,&d,&CommunicationDialog::testComplete);

    d.exec();
}

void MainWindow::updatePulseLeds(const PulseGenConfig cc)
{
    for(int i=0; i<d_ledList.size() && i < cc.size(); i++)
    {
        d_ledList.at(i).first->setText(cc.at(i).channelName);
        d_ledList.at(i).second->setState(cc.at(i).enabled);
    }
}

void MainWindow::updatePulseLed(int index, BlackChirp::PulseSetting s, QVariant val)
{
    if(index < 0 || index >= d_ledList.size())
        return;

    switch(s) {
    case BlackChirp::PulseNameSetting:
        d_ledList.at(index).first->setText(val.toString());
        break;
    case BlackChirp::PulseEnabledSetting:
        d_ledList.at(index).second->setState(val.toBool());
        break;
    default:
        break;
    }
}

void MainWindow::updateFlow(int ch, double val)
{
    if(ch < 0 || ch >= d_flowWidgets.size())
        return;

    d_flowWidgets.at(ch).displayBox->setValue(val);
}

void MainWindow::updateFlowName(int ch, QString name)
{
    if(ch < 0 || ch >= d_flowWidgets.size())
        return;

    if(name.isEmpty())
        d_flowWidgets.at(ch).nameLabel->setText(QString("Ch%1").arg(ch+1));
    else
    {
        d_flowWidgets.at(ch).nameLabel->setText(name.mid(0,9));
        d_flowWidgets.at(ch).nameEdit->blockSignals(true);
        d_flowWidgets.at(ch).nameEdit->setText(name);
        d_flowWidgets.at(ch).nameEdit->blockSignals(false);
    }
}

void MainWindow::updateFlowSetpoint(int ch, double val)
{
    if(ch < 0 || ch >= d_flowWidgets.size())
        return;

    d_flowWidgets.at(ch).controlBox->blockSignals(true);
    d_flowWidgets.at(ch).controlBox->setValue(val);
    d_flowWidgets.at(ch).controlBox->blockSignals(false);

    if(qFuzzyCompare(1.0,val+1.0))
        d_flowWidgets.at(ch).led->setState(false);
    else
        d_flowWidgets.at(ch).led->setState(true);
}

void MainWindow::updatePressureSetpoint(double val)
{
    ui->pressureControlBox->blockSignals(true);
    ui->pressureControlBox->setValue(val);
    ui->pressureControlBox->blockSignals(false);
}

void MainWindow::updatePressureControl(bool en)
{
    ui->pressureControlButton->setChecked(en);
    ui->pressureLed->setState(en);
}

void MainWindow::setLogIcon(BlackChirp::LogMessageCode c)
{
    if(ui->tabWidget->currentWidget() != ui->logTab)
    {
        switch(c) {
        case BlackChirp::LogWarning:
            if(d_logIcon != BlackChirp::LogError)
            {
                ui->tabWidget->setTabIcon(ui->tabWidget->indexOf(ui->logTab),QIcon(QString(":/icons/warning.png")));
                d_logIcon = c;
            }
            break;
        case BlackChirp::LogError:
            ui->tabWidget->setTabIcon(ui->tabWidget->indexOf(ui->logTab),QIcon(QString(":/icons/error.png")));
            d_logIcon = c;
            break;
        default:
            d_logIcon = c;
            ui->tabWidget->setTabIcon(ui->tabWidget->indexOf(ui->logTab),QIcon());
            break;
        }
    }
    else
    {
        d_logIcon = BlackChirp::LogNormal;
        ui->tabWidget->setTabIcon(ui->tabWidget->indexOf(ui->logTab),QIcon());
    }
}

void MainWindow::sleep(bool s)
{
    if(d_state == Acquiring)
    {
        QMessageBox mb(this);
        mb.setWindowTitle(QString("Sleep when complete"));
        mb.setText(QString("BlackChirp will sleep when the current experiment (or batch) is complete."));
        mb.addButton(QString("Cancel Sleep"),QMessageBox::RejectRole);
        connect(this,&MainWindow::checkSleep,&mb,&QMessageBox::accept);

        int ret = mb.exec();
        if(ret == QDialog::Accepted)
        {
            QMetaObject::invokeMethod(p_hwm,"sleep",Q_ARG(bool,true));
            configureUi(Asleep);
            ui->actionSleep->blockSignals(true);
            ui->actionSleep->setChecked(true);
            ui->actionSleep->blockSignals(false);
            QMessageBox::information(this,QString("BlackChirp Asleep"),QString("The instrument is asleep. Press the sleep button to re-activate it."),QMessageBox::Ok);
        }
        else
        {
            ui->actionSleep->blockSignals(true);
            ui->actionSleep->setChecked(false);
            ui->actionSleep->blockSignals(false);
        }
    }
    else
    {
        QMetaObject::invokeMethod(p_hwm,"sleep",Q_ARG(bool,s));
        if(s)
        {
            configureUi(Asleep);
            QMessageBox::information(this,QString("BlackChirp Asleep"),QString("The instrument is asleep. Press the sleep button to re-activate it."),QMessageBox::Ok);
        }
        else
            configureUi(Idle);
    }

}

void MainWindow::viewExperiment()
{
    QDialog d(this);
    d.setWindowTitle(QString("View experiment"));
    QVBoxLayout *vbl = new QVBoxLayout;
    QFormLayout *fl = new QFormLayout;
    QHBoxLayout *hl = new QHBoxLayout;

    QSpinBox *numBox = new QSpinBox(&d);

    fl->addRow(QString("Experiment Number"),numBox);

    QCheckBox *pathBox = new QCheckBox(QString("Specify path"),&d);
    fl->addRow(pathBox);

    QLineEdit *pathEdit = new QLineEdit(&d);
    QToolButton *browseButton = new QToolButton(&d);
    browseButton->setIcon(QIcon(QString(":/icons/view.png")));

    connect(browseButton,&QToolButton::clicked,[=](){
        QString path = QFileDialog::getExistingDirectory(this,QString("Select experiment directory"),QString("~"));
        if(!path.isEmpty())
            pathEdit->setText(path);
    });

    hl->addWidget(pathEdit,1);
    hl->addWidget(browseButton,0);
    fl->addRow(hl);

    int lastCompletedExperiment = ui->exptSpinBox->value();
    if(d_batchThread->isRunning() && d_currentExptNum == lastCompletedExperiment)
        lastCompletedExperiment--;

    if(lastCompletedExperiment < 1)
    {
        numBox->setRange(0,__INT_MAX__);
        numBox->setSpecialValueText(QString("Select..."));
        numBox->setEnabled(true);
        pathBox->setChecked(true);
        pathBox->setEnabled(true);
    }
    else
    {
        numBox->setRange(1,lastCompletedExperiment);
        numBox->setValue(lastCompletedExperiment);
        pathBox->setChecked(false);
        pathEdit->setEnabled(false);
        browseButton->setEnabled(false);
    }

    connect(pathBox,&QCheckBox::toggled,[=](bool checked){
       if(checked)
       {
           pathEdit->setEnabled(true);
           browseButton->setEnabled(true);
           numBox->setRange(1,__INT_MAX__);
       }
       else
       {
           numBox->setRange(1,lastCompletedExperiment);
           pathEdit->clear();
           pathEdit->setEnabled(false);
           browseButton->setEnabled(false);
       }
    });

    QDialogButtonBox *bb = new QDialogButtonBox(QDialogButtonBox::Open|QDialogButtonBox::Cancel,&d);

    connect(bb->button(QDialogButtonBox::Open),&QPushButton::clicked,&d,&QDialog::accept);
    connect(bb->button(QDialogButtonBox::Cancel),&QPushButton::clicked,&d,&QDialog::reject);

    vbl->addLayout(fl);
    vbl->addWidget(bb);

    d.setLayout(vbl);

    if(d.exec() == QDialog::Accepted)
    {
        QString path = QString("");
        if(pathBox->isChecked())
        {
            path = pathEdit->text();
            if(path.isEmpty())
            {
                QMessageBox::critical(this,QString("Load error"),QString("Cannot open experiment with an empty path."),QMessageBox::Ok);
                return;
            }

            QDir dir(path);
            if(!dir.exists())
            {
                QMessageBox::critical(this,QString("Load error"),QString("The directory %1 does not exist. Could not load experiment.").arg(dir.absolutePath()),QMessageBox::Ok);
                return;
            }
        }

        int num = numBox->value();
        if(num < 1)
        {
            QMessageBox::critical(this,QString("Load error"),QString("Cannot open an experiment numbered below 1. (You chose %1)").arg(num),QMessageBox::Ok);
            return;
        }

        ExperimentViewWidget *evw = new ExperimentViewWidget(num,path);
        connect(this,&MainWindow::closing,evw,&ExperimentViewWidget::close);
        connect(ui->ftViewWidget,&FtmwViewWidget::finalized,evw,&ExperimentViewWidget::ftmwFinalized);
        connect(evw,&ExperimentViewWidget::notifyUiFinalized,ui->ftViewWidget,&FtmwViewWidget::snapshotsFinalizedUpdateUi);
        evw->show();
        evw->raise();
    }
}

#ifdef BC_PCONTROLLER
void MainWindow::configPController(bool readOnly)
{
    QHBoxLayout *hbl = new QHBoxLayout;
    QLabel *cplabel = new QLabel(QString("Pressure"));
    cplabel->setAlignment(Qt::AlignRight);
    QDoubleSpinBox *cpbox = new QDoubleSpinBox;

    SettingsStorage s(BC::Key::pController,SettingsStorage::Hardware);


    cpbox->setMinimum(s.get<double>(BC::Key::pControllerMin,-1.0));
    cpbox->setMaximum(s.get<double>(BC::Key::pControllerMax,20.0));
    cpbox->setDecimals(s.get<int>(BC::Key::pControllerDecimals,4));
    cpbox->setSuffix(QString(" ")+s.get<QString>(BC::Key::pControllerUnits,"Torr"));

    cpbox->setReadOnly(true);
    cpbox->setButtonSymbols(QAbstractSpinBox::NoButtons);
    cpbox->setKeyboardTracking(false);
    cpbox->blockSignals(true);

    connect(p_hwm,&HardwareManager::pressureUpdate,cpbox,&QDoubleSpinBox::setValue);

    hbl->addWidget(cplabel,0);
    hbl->addWidget(cpbox,1);
    if(!readOnly)
    {
        Led *pcled = new Led();
        hbl->addWidget(pcled,0);

        QGroupBox *pcBox = new QGroupBox(QString("Chamber Pressure Control"));
        QHBoxLayout *hbl2 = new QHBoxLayout;

        QLabel *psLabel = new QLabel("Pressure Setpoint");
        psLabel->setAlignment(Qt::AlignRight);

        QDoubleSpinBox *pSetpointBox = new QDoubleSpinBox;
        pSetpointBox->setMinimum(cpbox->minimum());
        pSetpointBox->setMaximum(cpbox->maximum());
        pSetpointBox->setDecimals(cpbox->decimals());
        pSetpointBox->setSuffix(cpbox->suffix());

        pSetpointBox->setSingleStep(qAbs(pSetpointBox->maximum() - pSetpointBox->minimum())/100.0);
        pSetpointBox->setKeyboardTracking(false);
//        pSetpointBox->setEnabled(false);

        QPushButton *pControlButton = new QPushButton("Off");
        pControlButton->setCheckable(true);
        pControlButton->setChecked(false);

        connect(p_hwm,&HardwareManager::pressureSetpointUpdate,[=](double val){
           pSetpointBox->blockSignals(true);
           pSetpointBox->setValue(val);
           pSetpointBox->blockSignals(false);
        });
        connect(pSetpointBox,static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),p_hwm,&HardwareManager::setPressureSetpoint);
        connect(pControlButton,&QPushButton::toggled,p_hwm,&HardwareManager::setPressureControlMode);
        connect(p_hwm,&HardwareManager::pressureControlMode,[=](bool en){
            pControlButton->blockSignals(true);
            if(en)
            {
                pControlButton->setText(QString("On"));
            }
            else
            {
                pControlButton->setText(QString("Off"));
            }
            pControlButton->setChecked(en);
            pcled->setState(en);
            pControlButton->blockSignals(false);
        });

        hbl2->addWidget(psLabel,0);
        hbl2->addWidget(pSetpointBox,1);
        hbl2->addWidget(pControlButton,0);

        QHBoxLayout *hbl3 = new QHBoxLayout;

//        QLabel *vpLabel = new QLabel("Valve Position");
//        vpLabel->setAlignment(Qt::AlignRight);
//        QLabel *vpvLabel = new QLabel("");
//        vpLabel->setAlignment(Qt::AlignRight);

        QPushButton *pOpenButton = new QPushButton("Open");
        QPushButton *pCloseButton = new QPushButton("Close");
        connect(pOpenButton,&QPushButton::clicked,p_hwm,&HardwareManager::openGateValve);
        connect(pCloseButton,&QPushButton::clicked,p_hwm,&HardwareManager::closeGateValve);
        hbl3->addWidget(pOpenButton,0);
        hbl3->addWidget(pCloseButton,0);

        QVBoxLayout *vbl = new QVBoxLayout;

        vbl->addLayout(hbl2);
        vbl->addLayout(hbl3);

        pcBox->setLayout(vbl);

        ui->gasControlLayout->addWidget(pcBox,0);
    }

    QGroupBox *pgb = new QGroupBox(QString("Chamber Status"));
    pgb->setLayout(hbl);

    p_pcBox = pgb;
    ui->instrumentStatusLayout->insertWidget(2,pgb,0);
}
#endif

void MainWindow::configureUi(MainWindow::ProgramState s)
{
    d_state = s;
    if(!d_hardwareConnected)
        s = Disconnected;

    switch(s)
    {
    case Asleep:
        ui->actionAbort->setEnabled(false);
        ui->actionPause->setEnabled(false);
        ui->actionResume->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(false);
        ui->actionQuick_Experiment->setEnabled(false);
        ui->actionStart_Sequence->setEnabled(false);
        ui->actionCommunication->setEnabled(false);
        ui->actionTest_All_Connections->setEnabled(false);
        ui->gasControlBox->setEnabled(false);
        ui->pulseConfigWidget->setEnabled(false);
        ui->actionSleep->setEnabled(true);
#ifdef BC_LIF
        p_lifControlWidget->setEnabled(false);
#endif
        if(p_pcBox)
            p_pcBox->setEnabled(false);
        break;
    case Disconnected:
        ui->actionAbort->setEnabled(false);
        ui->actionPause->setEnabled(false);
        ui->actionResume->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(false);
        ui->actionQuick_Experiment->setEnabled(false);
        ui->actionStart_Sequence->setEnabled(false);
        ui->actionCommunication->setEnabled(true);
        ui->actionTest_All_Connections->setEnabled(true);
        ui->gasControlBox->setEnabled(false);
        ui->pulseConfigWidget->setEnabled(false);
        ui->actionSleep->setEnabled(false);
#ifdef BC_LIF
        p_lifControlWidget->setEnabled(false);
#endif
        if(p_pcBox)
            p_pcBox->setEnabled(false);
        break;
    case Paused:
        ui->actionAbort->setEnabled(true);
        ui->actionPause->setEnabled(false);
        ui->actionResume->setEnabled(true);
        ui->actionStart_Experiment->setEnabled(false);
        ui->actionQuick_Experiment->setEnabled(false);
        ui->actionStart_Sequence->setEnabled(false);
        ui->actionCommunication->setEnabled(false);
        ui->actionTest_All_Connections->setEnabled(false);
        ui->gasControlBox->setEnabled(false);
        ui->pulseConfigWidget->setEnabled(false);
        ui->actionSleep->setEnabled(false);
#ifdef BC_LIF
        p_lifControlWidget->setEnabled(false);
#endif
        if(p_pcBox)
            p_pcBox->setEnabled(false);
        break;
    case Acquiring:
        ui->actionAbort->setEnabled(true);
        ui->actionPause->setEnabled(true);
        ui->actionResume->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(false);
        ui->actionQuick_Experiment->setEnabled(false);
        ui->actionStart_Sequence->setEnabled(false);
        ui->actionCommunication->setEnabled(false);
        ui->actionTest_All_Connections->setEnabled(false);
        ui->gasControlBox->setEnabled(false);
        ui->pulseConfigWidget->setEnabled(false);
        ui->actionSleep->setEnabled(true);
#ifdef BC_LIF
        p_lifControlWidget->setEnabled(false);
#endif
        if(p_pcBox)
            p_pcBox->setEnabled(false);
        break;
    case Peaking:
        ui->actionAbort->setEnabled(true);
        ui->actionPause->setEnabled(false);
        ui->actionResume->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(false);
        ui->actionQuick_Experiment->setEnabled(false);
        ui->actionStart_Sequence->setEnabled(false);
        ui->actionCommunication->setEnabled(false);
        ui->actionTest_All_Connections->setEnabled(false);
        ui->gasControlBox->setEnabled(true);
        ui->pulseConfigWidget->setEnabled(true);
        ui->actionSleep->setEnabled(false);
#ifdef BC_LIF
        p_lifControlWidget->setEnabled(true);
#endif
        if(p_pcBox)
            p_pcBox->setEnabled(true);
        break;
    case Idle:
    default:
        ui->actionAbort->setEnabled(false);
        ui->actionPause->setEnabled(false);
        ui->actionResume->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(true);
        ui->actionQuick_Experiment->setEnabled(d_oneExptDone);
        ui->actionStart_Sequence->setEnabled(true);
        ui->actionCommunication->setEnabled(true);
        ui->actionTest_All_Connections->setEnabled(true);
        ui->gasControlBox->setEnabled(true);
        ui->pulseConfigWidget->setEnabled(true);
        ui->actionSleep->setEnabled(true);
#ifdef BC_LIF
        p_lifControlWidget->setEnabled(true);
#endif
        if(p_pcBox)
            p_pcBox->setEnabled(true);
        break;
    }
}

void MainWindow::startBatch(BatchManager *bm)
{
    connect(d_batchThread,&QThread::started,bm,&BatchManager::beginNextExperiment);
    connect(bm,&BatchManager::statusMessage,this,&MainWindow::statusMessage);
    connect(bm,&BatchManager::logMessage,p_lh,&LogHandler::logMessage);
    connect(bm,&BatchManager::beginExperiment,p_lh,&LogHandler::endExperimentLog);
    connect(bm,&BatchManager::beginExperiment,p_hwm,&HardwareManager::initializeExperiment);
    connect(p_am,&AcquisitionManager::experimentComplete,bm,&BatchManager::experimentComplete);
    connect(p_am,&AcquisitionManager::experimentComplete,ui->ftViewWidget,&FtmwViewWidget::experimentComplete);
    connect(ui->actionAbort,&QAction::triggered,bm,&BatchManager::abort);
    connect(bm,&BatchManager::batchComplete,this,&MainWindow::batchComplete);
    connect(bm,&BatchManager::batchComplete,this,&MainWindow::checkSleep);
    connect(bm,&BatchManager::batchComplete,d_batchThread,&QThread::quit);
    connect(bm,&BatchManager::batchComplete,p_lh,&LogHandler::endExperimentLog);
    connect(d_batchThread,&QThread::finished,bm,&BatchManager::deleteLater);

    connect(p_hwm,&HardwareManager::timeData,ui->trackingViewWidget,&TrackingViewWidget::pointUpdated,Qt::UniqueConnection);
    connect(p_am,&AcquisitionManager::timeData,ui->trackingViewWidget,&TrackingViewWidget::pointUpdated,Qt::UniqueConnection);
    connect(p_hwm,&HardwareManager::abortAcquisition,p_am,&AcquisitionManager::abort,Qt::UniqueConnection);

    ui->trackingViewWidget->initializeForExperiment();
    configureUi(Acquiring);
    bm->moveToThread(d_batchThread);
    d_batchThread->start();
}

FlowConfig MainWindow::getFlowConfig()
{
    FlowConfig cfg;
    cfg.setPressureControlMode(ui->pressureControlButton->isChecked());
    cfg.setPressureSetpoint(ui->pressureControlBox->value());
    for(int i=0; i<d_flowWidgets.size(); i++)
        cfg.add(d_flowWidgets.at(i).controlBox->value(),d_flowWidgets.at(i).nameEdit->text());

    return cfg;
}

void MainWindow::closeEvent(QCloseEvent *ev)
{
    if(d_batchThread->isRunning())
        ev->ignore();
    else
    {
        ev->accept();
        emit closing();
    }
}
