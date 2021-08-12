#include "mainwindow.h"
#include "mainwindow_ui.h"

#include <QThread>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QCloseEvent>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QMessageBox>
#include <QCheckBox>
#include <QToolButton>
#include <QFileDialog>
#include <QDir>

#include <gui/dialog/communicationdialog.h>
#include <gui/dialog/hwdialog.h>
#include <gui/widget/digitizerconfigwidget.h>
#include <gui/widget/rfconfigwidget.h>
#include <gui/wizard/experimentwizard.h>
#include <gui/widget/pulseconfigwidget.h>
#include <gui/widget/gascontrolwidget.h>
#include <gui/widget/gasflowdisplaywidget.h>
#include <gui/widget/pulsestatusbox.h>
#include <data/loghandler.h>
#include <hardware/core/hardwaremanager.h>
#include <acquisition/acquisitionmanager.h>
#include <acquisition/batch/batchmanager.h>
#include <acquisition/batch/batchsingle.h>
#include <acquisition/batch/batchsequence.h>
#include <gui/widget/led.h>
#include <gui/widget/experimentviewwidget.h>
#include <gui/dialog/quickexptdialog.h>
#include <gui/dialog/batchsequencedialog.h>
#include <gui/widget/clockdisplaywidget.h>
#include <gui/widget/gascontrolwidget.h>
#include <gui/widget/pressurestatusbox.h>
#include <gui/widget/pressurecontrolwidget.h>

#ifdef BC_LIF
#include <modules/lif/gui/lifdisplaywidget.h>
#include <modules/lif/gui/lifcontrolwidget.h>
#endif

#ifdef BC_MOTOR
#include <modules/motor/gui/motordisplaywidget.h>
#include <modules/motor/gui/motorstatuswidget.h>
#endif

#include <hardware/optional/tempcontroller/temperaturecontroller.h>
#include <hardware/optional/pressurecontroller/pressurecontroller.h>
#include <hardware/optional/flowcontroller/flowcontroller.h>
#include <hardware/optional/pulsegenerator/pulsegenerator.h>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow), d_hardwareConnected(false), d_oneExptDone(false), d_state(Idle), d_logCount(0), d_logIcon(BlackChirp::LogNormal), d_currentExptNum(0)
{
    p_hwm = new HardwareManager();
    auto hwl = p_hwm->currentHardware();

    qRegisterMetaType<QwtPlot::Axis>("QwtPlot::Axis");

    ui->setupUi(this);

    p_lh = new LogHandler();
    connect(this,&MainWindow::logMessage,p_lh,&LogHandler::logMessage);
    connect(p_lh,&LogHandler::sendLogMessage,ui->logTextEdit,&QTextEdit::append);
    connect(p_lh,&LogHandler::iconUpdate,this,&MainWindow::setLogIcon);
    connect(ui->ftViewWidget,&FtmwViewWidget::experimentLogMessage,p_lh,&LogHandler::experimentLogMessage);
    connect(ui->mainTabWidget,&QTabWidget::currentChanged,[=](int i) {
        if(i == ui->mainTabWidget->indexOf(ui->logTab))
        {
            setLogIcon(BlackChirp::LogNormal);
            if(d_logCount > 0)
            {
                d_logCount = 0;
                ui->mainTabWidget->setTabText(ui->mainTabWidget->indexOf(ui->logTab),QString("Log"));
            }
        }
    });
    connect(p_lh,&LogHandler::sendLogMessage,this,[=](){
        if(ui->mainTabWidget->currentWidget() != ui->logTab)
        {
            d_logCount++;
            ui->mainTabWidget->setTabText(ui->mainTabWidget->indexOf(ui->logTab),QString("Log (%1)").arg(d_logCount));
        }
    });

    QThread *lhThread = new QThread(this);
    lhThread->setObjectName("LogHandlerThread");
    connect(lhThread,&QThread::finished,p_lh,&LogHandler::deleteLater);
    p_lh->moveToThread(lhThread);
    d_threadObjectList.append(qMakePair(lhThread,p_lh));
    lhThread->start();

    connect(p_hwm,&HardwareManager::logMessage,p_lh,&LogHandler::logMessage);
    connect(p_hwm,&HardwareManager::statusMessage,ui->statusBar,&QStatusBar::showMessage);
    connect(p_hwm,&HardwareManager::allHardwareConnected,this,&MainWindow::hardwareInitialized);

    connect(p_hwm,&HardwareManager::clockFrequencyUpdate,ui->clockWidget,&ClockDisplayWidget::updateFrequency);


    for(auto it = hwl.cbegin(); it != hwl.cend(); ++it)
    {
        auto key = it->first;
        auto act = ui->menuHardware->addAction(key);

        if(key == BC::Key::Flow::flowController)
        {
            auto w = new GasFlowDisplayBox;
            w->setObjectName(key);
            ui->instrumentStatusLayout->insertWidget(3,w,0);
            connect(p_hwm,&HardwareManager::flowUpdate,w,&GasFlowDisplayBox::updateFlow);
            connect(p_hwm,&HardwareManager::flowSetpointUpdate,w,&GasFlowDisplayBox::updateFlowSetpoint);
            connect(p_hwm,&HardwareManager::gasPressureUpdate,w,&GasFlowDisplayBox::updatePressure);
            connect(p_hwm,&HardwareManager::gasPressureControlMode,w,&GasFlowDisplayBox::updatePressureControl);

            connect(act,&QAction::triggered,[this,w,key]{

                if(isDialogOpen(key))
                    return;

                auto gcw = new GasControlWidget;
                auto fc = p_hwm->getFlowConfig();
                gcw->initialize(fc);
                connect(p_hwm,&HardwareManager::flowSetpointUpdate,gcw,&GasControlWidget::updateGasSetpoint);
                connect(p_hwm,&HardwareManager::gasPressureSetpointUpdate,gcw,&GasControlWidget::updatePressureSetpoint);
                connect(p_hwm,&HardwareManager::gasPressureControlMode,gcw,&GasControlWidget::updatePressureControl);
                connect(gcw,&GasControlWidget::pressureControlUpdate,p_hwm,&HardwareManager::setGasPressureControlMode);
                connect(gcw,&GasControlWidget::pressureSetpointUpdate,p_hwm,&HardwareManager::setGasPressureSetpoint);
                connect(gcw,&GasControlWidget::gasSetpointUpdate,p_hwm,&HardwareManager::setFlowSetpoint);
                connect(gcw,&GasControlWidget::nameUpdate,w,&GasFlowDisplayBox::updateFlowName);
                connect(gcw,&GasControlWidget::nameUpdate,p_hwm,&HardwareManager::setFlowChannelName);

                auto d = createHWDialog(key,gcw);
                connect(d,&QDialog::accepted,w,&GasFlowDisplayBox::applySettings);

            });
        }
        else if(key == BC::Key::PController::key)
        {
            auto psb = new PressureStatusBox;
            psb->setObjectName(key);
            ui->instrumentStatusLayout->insertWidget(3,psb,0);
            connect(p_hwm,&HardwareManager::pressureUpdate,psb,&PressureStatusBox::pressureUpdate);
            connect(p_hwm,&HardwareManager::pressureControlMode,psb,&PressureStatusBox::pressureControlUpdate);

            connect(act,&QAction::triggered,[this,psb,key](){

                if(isDialogOpen(key))
                    return;

                auto pcw = new PressureControlWidget;
                auto pc = p_hwm->getPressureControllerConfig();
                pcw->initialize(pc);
                connect(p_hwm,&HardwareManager::pressureSetpointUpdate,pcw,&PressureControlWidget::pressureSetpointUpdate);
                connect(p_hwm,&HardwareManager::pressureControlMode,pcw,&PressureControlWidget::pressureControlModeUpdate);
                connect(pcw,&PressureControlWidget::setpointChanged,p_hwm,&HardwareManager::setPressureSetpoint);
                connect(pcw,&PressureControlWidget::pressureControlModeChanged,p_hwm,&HardwareManager::setPressureControlMode);
                connect(pcw,&PressureControlWidget::valveOpen,p_hwm,&HardwareManager::openGateValve);
                connect(pcw,&PressureControlWidget::valveClose,p_hwm,&HardwareManager::closeGateValve);

                auto d = createHWDialog(key,pcw);
                connect(d,&QDialog::accepted,psb,&PressureStatusBox::updateFromSettings);
            });
        }
        else if(key == BC::Key::PGen::key)
        {
            auto psb = new PulseStatusBox;
            psb->setObjectName(key);
            ui->instrumentStatusLayout->insertWidget(3,psb,0);
            connect(p_hwm,&HardwareManager::pGenConfigUpdate,psb,&PulseStatusBox::updatePulseLeds);
            connect(p_hwm,&HardwareManager::pGenSettingUpdate,psb,&PulseStatusBox::updatePulseLed);

            connect(act,&QAction::triggered,[this,psb,key]{
               if(isDialogOpen(key))
                   return;

               auto pcw = new PulseConfigWidget;
               auto pc = p_hwm->getPGenConfig();
               pcw->setFromConfig(pc);

               connect(p_hwm,&HardwareManager::pGenConfigUpdate,pcw,&PulseConfigWidget::setFromConfig);
               connect(p_hwm,&HardwareManager::pGenSettingUpdate,pcw,&PulseConfigWidget::newSetting);
               connect(p_hwm,&HardwareManager::pGenRepRateUpdate,pcw,&PulseConfigWidget::newRepRate);
               connect(pcw,&PulseConfigWidget::changeSetting,p_hwm,&HardwareManager::setPGenSetting);
               connect(pcw,&PulseConfigWidget::changeRepRate,p_hwm,&HardwareManager::setPGenRepRate);

               auto d = createHWDialog(key,pcw);
               connect(d,&QDialog::accepted,psb,&PulseStatusBox::updateFromSettings);
            });

        }
        else
        {
            connect(act,&QAction::triggered,[this,key](){
                if(isDialogOpen(key))
                    return;

                createHWDialog(key);
            });
        }
    }

    QThread *hwmThread = new QThread(this);
    hwmThread->setObjectName("HardwareManagerThread");
    connect(hwmThread,&QThread::started,p_hwm,&HardwareManager::initialize);
    connect(hwmThread,&QThread::finished,p_hwm,&HardwareManager::deleteLater);
    p_hwm->moveToThread(hwmThread);
    d_threadObjectList.append(qMakePair(hwmThread,p_hwm));



    p_am = new AcquisitionManager();
    connect(p_am,&AcquisitionManager::logMessage,p_lh,&LogHandler::logMessage);
    connect(p_am,&AcquisitionManager::statusMessage,ui->statusBar,&QStatusBar::showMessage);
    connect(p_am,&AcquisitionManager::ftmwUpdateProgress,ui->ftmwProgressBar,&QProgressBar::setValue);
    connect(ui->pauseButton,&QToolButton::triggered,p_am,&AcquisitionManager::pause);
    connect(ui->resumeButton,&QToolButton::triggered,p_am,&AcquisitionManager::resume);
    connect(ui->abortButton,&QToolButton::triggered,p_am,&AcquisitionManager::abort);
    connect(p_am,&AcquisitionManager::snapshotComplete,ui->ftViewWidget,&FtmwViewWidget::snapshotTaken);
    connect(p_am,&AcquisitionManager::experimentComplete,ui->ftViewWidget,&FtmwViewWidget::experimentComplete);

    QThread *amThread = new QThread(this);
    amThread->setObjectName("AcquisitionManagerThread");
    connect(amThread,&QThread::finished,p_am,&AcquisitionManager::deleteLater);
    p_am->moveToThread(amThread);
    d_threadObjectList.append(qMakePair(amThread,p_am));


    connect(p_hwm,&HardwareManager::experimentInitialized,this,&MainWindow::experimentInitialized);
    connect(p_hwm,&HardwareManager::ftmwScopeShotAcquired,p_am,&AcquisitionManager::processFtmwScopeShot);
    connect(p_am,&AcquisitionManager::newClockSettings,p_hwm,&HardwareManager::setClocks);
    connect(p_hwm,&HardwareManager::allClocksReady,p_am,&AcquisitionManager::clockSettingsComplete);
    connect(p_am,&AcquisitionManager::beginAcquisition,p_hwm,&HardwareManager::beginAcquisition);
    connect(p_am,&AcquisitionManager::endAcquisition,p_hwm,&HardwareManager::endAcquisition);
    connect(p_am,&AcquisitionManager::auxDataSignal,p_hwm,&HardwareManager::getAuxData);
    connect(p_hwm,&HardwareManager::auxData,p_am,&AcquisitionManager::processAuxData);
    connect(p_hwm,&HardwareManager::validationData,p_am,&AcquisitionManager::processValidationData);
    connect(p_hwm,&HardwareManager::rollingData,ui->rollingDataViewWidget,&RollingDataWidget::pointUpdated);


    connect(this,&MainWindow::startInit,[=](){
        hwmThread->start();
        amThread->start();
    });

    p_batchThread = new QThread(this);
    p_batchThread->setObjectName("BatchManagerThread");

    connect(ui->actionStart_Experiment,&QAction::triggered,this,&MainWindow::startExperiment);
    connect(ui->actionQuick_Experiment,&QAction::triggered,this,&MainWindow::quickStart);
    connect(ui->actionStart_Sequence,&QAction::triggered,this,&MainWindow::startSequence);
    connect(ui->pauseButton,&QToolButton::triggered,this,&MainWindow::pauseUi);
    connect(ui->resumeButton,&QToolButton::triggered,this,&MainWindow::resumeUi);
    connect(ui->actionCommunication,&QAction::triggered,this,&MainWindow::launchCommunicationDialog);
    connect(ui->actionRfConfig,&QAction::triggered,this,&MainWindow::launchRfConfigDialog);
    connect(ui->action_AuxGraphs,&QAction::triggered,ui->auxDataViewWidget,&AuxDataViewWidget::changeNumPlots);
    connect(ui->actionAutoscale_Aux,&QAction::triggered,ui->auxDataViewWidget,&AuxDataViewWidget::autoScaleAll);
    connect(ui->action_RollingGraphs,&QAction::triggered,ui->rollingDataViewWidget,&RollingDataWidget::changeNumPlots);
    connect(ui->actionAutoscale_Rolling,&QAction::triggered,ui->rollingDataViewWidget,&RollingDataWidget::autoScaleAll);
    connect(ui->sleepButton,&QToolButton::toggled,this,&MainWindow::sleep);
    connect(ui->actionTest_All_Connections,&QAction::triggered,p_hwm,&HardwareManager::testAll);
    connect(ui->actionView_Experiment,&QAction::triggered,this,&MainWindow::viewExperiment);
#ifdef BC_LIF
    p_lifDisplayWidget = new LifDisplayWidget(this);
    int lti = ui->mainTabWidget->insertTab(ui->mainTabWidget->indexOf(ui->trackingTab),p_lifDisplayWidget,QIcon(QString(":/icons/laser.png")),QString("LIF"));
    p_lifTab = ui->mainTabWidget->widget(lti);
    p_lifProgressBar = new QProgressBar(this);
    ui->instrumentStatusLayout->addWidget(new QLabel(QString("LIF Progress")),0,Qt::AlignCenter);
    ui->instrumentStatusLayout->addWidget(p_lifProgressBar);
    p_lifControlWidget = new LifControlWidget(this);
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
    connect(p_lifAction,&QAction::triggered,this,[=](){ ui->mainTabWidget->setCurrentWidget(p_lifTab); });

#endif

#ifdef BC_MOTOR
    p_motorDisplayWidget = new MotorDisplayWidget(this);
    int mti = ui->mainTabWidget->insertTab(ui->mainTabWidget->indexOf(ui->trackingTab),p_motorDisplayWidget,QIcon(QString(":/icons/motorscan.png")),QString("Motor"));
    p_motorTab = ui->mainTabWidget->widget(mti);

    p_motorViewAction = new QAction(QIcon(QString(":/icons/motorscan.png")),QString("Motor"),this);
    ui->menuView->insertAction(ui->actionLog,p_motorViewAction);
    connect(p_motorViewAction,&QAction::triggered,[=](){ ui->mainTabWidget->setCurrentWidget(p_motorTab);});

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
    delete ui;
}

void MainWindow::initializeHardware()
{
    ui->statusBar->showMessage(QString("Initializing hardware..."));
    emit startInit();
}

void MainWindow::startExperiment()
{
    if(p_batchThread->isRunning())
        return;

    ExperimentWizard wiz(-1,this);
    wiz.experiment->setPulseGenConfig(p_hwm->getPGenConfig());
    wiz.experiment->setFlowConfig(p_hwm->getFlowConfig());
    wiz.setValidationKeys(p_hwm->validationKeys());

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
    if(p_batchThread->isRunning())
        return;

    SettingsStorage s;
    int num = s.get(BC::Key::exptNum,0);
    QString path = s.get(BC::Key::savePath,QString(""));
    if(num < 1)
    {
        startExperiment();
        return;
    }

    std::shared_ptr<Experiment> e = std::make_shared<Experiment>(num,path,true);
#ifdef BC_LIF
    if(e.lifConfig().isEnabled())
    {
        LifConfig lc = e.lifConfig();
        lc = p_lifControlWidget->getSettings(lc);
        e.setLifConfig(lc);
    }
#endif
    e->setFlowConfig(p_hwm->getFlowConfig());
    e->setPulseGenConfig(p_hwm->getPGenConfig());

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
    if(p_batchThread->isRunning())
        return;

    BatchSequenceDialog d(this);
    d.setQuickExptEnabled(d_oneExptDone);
    int ret = d.exec();

    if(ret == QDialog::Rejected)
        return;

    std::shared_ptr<Experiment> exp;
    SettingsStorage s;
    int num = s.get(BC::Key::exptNum,0);
    QString path = s.get(BC::Key::savePath,QString(""));

    if(ret == d.quickCode)
    {
        exp = std::make_shared<Experiment>(num,path,true);
#ifdef BC_LIF
        if(e.lifConfig().isEnabled())
        {
            LifConfig lc = e.lifConfig();
            lc = p_lifControlWidget->getSettings(lc);
            e.setLifConfig(lc);
        }
#endif
        exp->setFlowConfig(p_hwm->getFlowConfig());
        exp->setPulseGenConfig(p_hwm->getPGenConfig());

        //create a popup summary of experiment.
        QuickExptDialog qd(exp,this);
        int qeret = qd.exec();

        if(qeret == QDialog::Accepted)
            ret = QDialog::Accepted;
        else if(qeret == qd.configureResult())
            ret = d.configureCode; //set ret to indicate that the experiment needs to be configured
        else if(qeret == QDialog::Rejected)
            return;
    }

    if(ret == d.configureCode)
    {
        ExperimentWizard wiz(-1,this);
        wiz.experiment->setPulseGenConfig(p_hwm->getPGenConfig());
        wiz.experiment->setFlowConfig(p_hwm->getFlowConfig());
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


    BatchSequence *bs = new BatchSequence(exp,d.numExperiments(),d.interval());
    startBatch(bs);

}

void MainWindow::batchComplete(bool aborted)
{
    disconnect(p_am,&AcquisitionManager::auxData,ui->auxDataViewWidget,&AuxDataViewWidget::pointUpdated);
    disconnect(p_hwm,&HardwareManager::abortAcquisition,p_am,&AcquisitionManager::abort);

#ifdef BC_LIF
    p_lifTab->setEnabled(true);
#endif

    if(aborted)
        ui->statusBar->showMessage(QString("Experiment aborted"));
    else
        ui->statusBar->showMessage(QString("Experiment complete"));

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

void MainWindow::experimentInitialized(std::shared_ptr<Experiment> exp)
{   
    if(!p_batchThread->isRunning())
        return;

    if(!exp->d_hardwareSuccess)
    {
        emit logMessage(exp->d_errorString,BlackChirp::LogError);
        p_batchThread->quit();
        configureUi(Idle);
        return;
    }

    if(!exp->initialize())
    {
        emit logMessage(QString("Could not initialize experiment."),BlackChirp::LogError);
        if(!exp->d_errorString.isEmpty())
            emit logMessage(exp->d_errorString,BlackChirp::LogError);
        p_batchThread->quit();
        configureUi(Idle);
        return;
    }

    if(exp->d_number > 0)
        ui->exptSpinBox->setValue(exp->d_number);

    d_currentExptNum = exp->d_number;

    ui->ftmwProgressBar->setValue(0);
    ui->ftViewWidget->prepareForExperiment(*exp);
    ui->auxDataViewWidget->initializeForExperiment();

    if(exp->ftmwEnabled())
	{
        if(exp->ftmwConfig()->indefinite())
            ui->ftmwProgressBar->setRange(0,0);
        else
            ui->ftmwProgressBar->setRange(0,1000);
	}
    else
    {
        ui->ftmwTab->setEnabled(false);
        ui->ftmwProgressBar->setRange(0,1);
        ui->ftmwProgressBar->setValue(1);
    }

#ifdef BC_LIF
#pragma message("Update LIF progress bar")
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

    if(!exp->isDummy())
    {
        if(p_lh->thread() == thread())
            p_lh->beginExperimentLog(exp->d_number,exp->d_startLogMessage);
        else
            QMetaObject::invokeMethod(p_lh,[this,exp](){p_lh->beginExperimentLog(exp->d_number,exp->d_startLogMessage);});
    }
    else
    {
        if(p_lh->thread() == thread())
            p_lh->logMessage(exp->d_startLogMessage,BlackChirp::LogHighlight);
        else
            emit logMessage(exp->d_startLogMessage,BlackChirp::LogHighlight);
    }

    QMetaObject::invokeMethod(p_am,[this,exp](){p_am->beginExperiment(exp);});


}

void MainWindow::hardwareInitialized(bool success)
{
	d_hardwareConnected = success;
    if(success)
        ui->statusBar->showMessage(QString("Hardware connected"));
    else
        ui->statusBar->showMessage(QString("Hardware error. See log for details."));
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

void MainWindow::launchCommunicationDialog(bool parent)
{
    QWidget *p = nullptr;
    if(parent)
        p = this;

    CommunicationDialog d(p);
    connect(&d,&CommunicationDialog::testConnection,p_hwm,&HardwareManager::testObjectConnection);
    connect(p_hwm,&HardwareManager::testComplete,&d,&CommunicationDialog::testComplete);

    d.exec();
}

void MainWindow::launchRfConfigDialog()
{
    auto d = new QDialog;
    auto w = new RfConfigWidget(d);
    RfConfig cfg;
    QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks;
    QMetaObject::invokeMethod(p_hwm,&HardwareManager::getClocks,Qt::BlockingQueuedConnection,&clocks);
    cfg.setCurrentClocks(clocks);
    w->setClocks(cfg);

    auto vbl = new QVBoxLayout;

    auto lbl = new QLabel("Settings will be applied when this dialog is closed with the Ok button.");
    lbl->setWordWrap(true);
    vbl->addWidget(lbl);
    vbl->addWidget(w);

    auto bb = new QDialogButtonBox(QDialogButtonBox::Ok|QDialogButtonBox::Cancel,d);
    connect(bb->button(QDialogButtonBox::Ok),&QPushButton::clicked,d,&QDialog::accept);
    connect(bb->button(QDialogButtonBox::Cancel),&QPushButton::clicked,d,&QDialog::reject);
    vbl->addWidget(bb);
    d->setLayout(vbl);

    connect(d,&QDialog::accepted,[this,w](){
        RfConfig rfc;
        w->toRfConfig(rfc);
        QMetaObject::invokeMethod(p_hwm,[rfc,this](){ p_hwm->configureClocks(rfc.getClocks());} );
    });

    d_openDialogs.insert({"RfConfig",d});
    d->show();

}

void MainWindow::setLogIcon(BlackChirp::LogMessageCode c)
{
    if(ui->mainTabWidget->currentWidget() != ui->logTab)
    {
        switch(c) {
        case BlackChirp::LogWarning:
            if(d_logIcon != BlackChirp::LogError)
            {
                ui->mainTabWidget->setTabIcon(ui->mainTabWidget->indexOf(ui->logTab),QIcon(QString(":/icons/warning.png")));
                d_logIcon = c;
            }
            break;
        case BlackChirp::LogError:
            ui->mainTabWidget->setTabIcon(ui->mainTabWidget->indexOf(ui->logTab),QIcon(QString(":/icons/error.png")));
            d_logIcon = c;
            break;
        default:
            d_logIcon = c;
            ui->mainTabWidget->setTabIcon(ui->mainTabWidget->indexOf(ui->logTab),QIcon());
            break;
        }
    }
    else
    {
        d_logIcon = BlackChirp::LogNormal;
        ui->mainTabWidget->setTabIcon(ui->mainTabWidget->indexOf(ui->logTab),QIcon());
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
            QMetaObject::invokeMethod(p_hwm,[this](){p_hwm->sleep(true);});
            configureUi(Asleep);
            ui->sleepButton->blockSignals(true);
            ui->sleepButton->setChecked(true);
            ui->sleepButton->blockSignals(false);
            QMessageBox::information(this,QString("BlackChirp Asleep"),QString("The instrument is asleep. Press the sleep button to re-activate it."),QMessageBox::Ok);
        }
        else
        {
            ui->sleepButton->blockSignals(true);
            ui->sleepButton->setChecked(false);
            ui->sleepButton->blockSignals(false);
        }
    }
    else
    {
        QMetaObject::invokeMethod(p_hwm,[this,s](){p_hwm->sleep(s);});
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
    if(p_batchThread->isRunning() && d_currentExptNum == lastCompletedExperiment)
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

bool MainWindow::isDialogOpen(const QString key)
{
    auto it = d_openDialogs.find(key);
    if(it != d_openDialogs.end())
    {
        auto d = it->second;
        d->setWindowState(Qt::WindowActive);
        d->raise();
        d->show();
        return true;
    }

    return false;
}

HWDialog *MainWindow::createHWDialog(const QString key, QWidget *controlWidget)
{
    auto out = new HWDialog(key,p_hwm->getForbiddenKeys(key),controlWidget);
    d_openDialogs.insert({key,out});
    auto hwm = p_hwm;
    connect(out,&HWDialog::accepted,[hwm,key](){
        QMetaObject::invokeMethod(hwm,[=](){ hwm->updateObjectSettings(key); });
    });
    connect(out,&HWDialog::destroyed,[this,key](){
        auto it = d_openDialogs.find(key);
        if(it != d_openDialogs.end())
            d_openDialogs.erase(it);
    });

    out->show();
    return out;
}

void MainWindow::configureUi(MainWindow::ProgramState s)
{
    d_state = s;
    if(!d_hardwareConnected)
        s = Disconnected;

    switch(s)
    {
    case Asleep:
        ui->abortButton->setEnabled(false);
        ui->pauseButton->setEnabled(false);
        ui->resumeButton->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(false);
        ui->actionQuick_Experiment->setEnabled(false);
        ui->actionStart_Sequence->setEnabled(false);
        ui->actionCommunication->setEnabled(false);
        ui->actionTest_All_Connections->setEnabled(false);
        ui->sleepButton->setEnabled(true);
#ifdef BC_LIF
        p_lifControlWidget->setEnabled(false);
#endif
        break;
    case Disconnected:
        ui->abortButton->setEnabled(false);
        ui->pauseButton->setEnabled(false);
        ui->resumeButton->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(false);
        ui->actionQuick_Experiment->setEnabled(false);
        ui->actionStart_Sequence->setEnabled(false);
        ui->actionCommunication->setEnabled(true);
        ui->actionTest_All_Connections->setEnabled(true);
        ui->sleepButton->setEnabled(false);
#ifdef BC_LIF
        p_lifControlWidget->setEnabled(false);
#endif
        break;
    case Paused:
        ui->abortButton->setEnabled(true);
        ui->pauseButton->setEnabled(false);
        ui->resumeButton->setEnabled(true);
        ui->actionStart_Experiment->setEnabled(false);
        ui->actionQuick_Experiment->setEnabled(false);
        ui->actionStart_Sequence->setEnabled(false);
        ui->actionCommunication->setEnabled(false);
        ui->actionTest_All_Connections->setEnabled(false);
        ui->sleepButton->setEnabled(false);
#ifdef BC_LIF
        p_lifControlWidget->setEnabled(false);
#endif
        break;
    case Acquiring:
        ui->abortButton->setEnabled(true);
        ui->pauseButton->setEnabled(true);
        ui->resumeButton->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(false);
        ui->actionQuick_Experiment->setEnabled(false);
        ui->actionStart_Sequence->setEnabled(false);
        ui->actionCommunication->setEnabled(false);
        ui->actionTest_All_Connections->setEnabled(false);
        ui->sleepButton->setEnabled(true);
#ifdef BC_LIF
        p_lifControlWidget->setEnabled(false);
#endif
        break;
    case Peaking:
        ui->abortButton->setEnabled(true);
        ui->pauseButton->setEnabled(false);
        ui->resumeButton->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(false);
        ui->actionQuick_Experiment->setEnabled(false);
        ui->actionStart_Sequence->setEnabled(false);
        ui->actionCommunication->setEnabled(false);
        ui->actionTest_All_Connections->setEnabled(false);
        ui->sleepButton->setEnabled(false);
#ifdef BC_LIF
        p_lifControlWidget->setEnabled(true);
#endif
        break;
    case Idle:
    default:
        ui->abortButton->setEnabled(false);
        ui->pauseButton->setEnabled(false);
        ui->resumeButton->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(true);
        ui->actionQuick_Experiment->setEnabled(d_oneExptDone);
        ui->actionStart_Sequence->setEnabled(true);
        ui->actionCommunication->setEnabled(true);
        ui->actionTest_All_Connections->setEnabled(true);
        ui->sleepButton->setEnabled(true);
#ifdef BC_LIF
        p_lifControlWidget->setEnabled(true);
#endif
        break;
    }
}

void MainWindow::startBatch(BatchManager *bm)
{
    connect(p_batchThread,&QThread::started,bm,&BatchManager::beginNextExperiment);
    connect(bm,&BatchManager::statusMessage,ui->statusBar,&QStatusBar::showMessage);
    connect(bm,&BatchManager::logMessage,p_lh,&LogHandler::logMessage);
    connect(bm,&BatchManager::beginExperiment,p_lh,&LogHandler::endExperimentLog);
    connect(bm,&BatchManager::beginExperiment,[this,bm](){p_hwm->initializeExperiment(bm->currentExperiment());});
    connect(p_am,&AcquisitionManager::experimentComplete,bm,&BatchManager::experimentComplete);
    connect(p_am,&AcquisitionManager::experimentComplete,ui->ftViewWidget,&FtmwViewWidget::experimentComplete);
    connect(ui->abortButton,&QToolButton::triggered,bm,&BatchManager::abort);
    connect(bm,&BatchManager::batchComplete,this,&MainWindow::batchComplete);
    connect(bm,&BatchManager::batchComplete,this,&MainWindow::checkSleep);
    connect(bm,&BatchManager::batchComplete,p_batchThread,&QThread::quit);
    connect(bm,&BatchManager::batchComplete,p_lh,&LogHandler::endExperimentLog);
    connect(p_batchThread,&QThread::finished,bm,&BatchManager::deleteLater);

    connect(p_am,&AcquisitionManager::auxData,ui->auxDataViewWidget,&AuxDataViewWidget::pointUpdated,Qt::UniqueConnection);
    connect(p_hwm,&HardwareManager::abortAcquisition,p_am,&AcquisitionManager::abort,Qt::UniqueConnection);

//    ui->trackingViewWidget->initializeForExperiment();
    configureUi(Acquiring);
    bm->moveToThread(p_batchThread);
    p_batchThread->start();
}

void MainWindow::closeEvent(QCloseEvent *ev)
{
    if(p_batchThread->isRunning())
        ev->ignore();
    else
    {
        emit closing();

        while(!d_threadObjectList.isEmpty())
        {
            QPair<QThread*,QObject*> p = d_threadObjectList.takeFirst();

            p.first->quit();
            p.first->wait();
        }

        ev->accept();
    }
}
