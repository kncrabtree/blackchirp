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
#include <QFontDialog>

#include <gui/dialog/communicationdialog.h>
#include <gui/dialog/hwdialog.h>
#include <gui/widget/digitizerconfigwidget.h>
#include <gui/widget/rfconfigwidget.h>
#include <gui/wizard/experimentwizard.h>
#include <gui/widget/pulseconfigwidget.h>
#include <gui/widget/gascontrolwidget.h>
#include <gui/widget/gasflowdisplaywidget.h>
#include <gui/widget/pulsestatusbox.h>
#include <gui/widget/temperaturestatusbox.h>
#include <gui/widget/temperaturecontrolwidget.h>
#include <data/loghandler.h>
#include <hardware/core/hardwaremanager.h>
#include <acquisition/acquisitionmanager.h>
#include <acquisition/batch/batchmanager.h>
#include <acquisition/batch/batchsingle.h>
#include <acquisition/batch/batchsequence.h>
#include <gui/widget/led.h>
#include <gui/dialog/bcsavepathdialog.h>
#include <gui/widget/experimentviewwidget.h>
#include <gui/dialog/quickexptdialog.h>
#include <gui/dialog/batchsequencedialog.h>
#include <gui/widget/clockdisplaybox.h>
#include <gui/widget/gascontrolwidget.h>
#include <gui/widget/pressurestatusbox.h>
#include <gui/widget/pressurecontrolwidget.h>

#ifdef BC_LIF
#include <modules/lif/gui/lifdisplaywidget.h>
#include <modules/lif/gui/lifcontrolwidget.h>
#include <modules/lif/gui/liflaserstatusbox.h>
#include <modules/lif/hardware/liflaser/liflaser.h>
#endif

#include <hardware/optional/tempcontroller/temperaturecontroller.h>
#include <hardware/optional/pressurecontroller/pressurecontroller.h>
#include <hardware/optional/flowcontroller/flowcontroller.h>
#include <hardware/optional/pulsegenerator/pulsegenerator.h>
#include <hardware/core/clock/fixedclock.h>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    p_hwm = new HardwareManager();
    d_hardware = p_hwm->currentHardware();

    qRegisterMetaType<QwtPlot::Axis>("QwtPlot::Axis");
#ifdef BC_LIF
    qRegisterMetaType<LifDigitizerConfig>("LifDigitizerConfig");
#endif

    ui->setupUi(this);
    ui->rollingDurationBox->setValue(ui->rollingDataViewWidget->historyDuration());
    connect(ui->rollingDurationBox,&SpinBoxWidgetAction::valueChanged,
            ui->rollingDataViewWidget,&RollingDataWidget::setHistoryDuration);

    ui->auxGraphsBox->setValue(ui->auxDataViewWidget->numPlots());
    connect(ui->auxGraphsBox,&SpinBoxWidgetAction::valueChanged,
            ui->auxDataViewWidget,&AuxDataViewWidget::changeNumPlots);

    ui->rollingGraphsBox->setValue(ui->rollingDataViewWidget->numPlots());
    connect(ui->rollingGraphsBox,&SpinBoxWidgetAction::valueChanged,
            ui->rollingDataViewWidget,&RollingDataWidget::changeNumPlots);

    connect(ui->fontAction,&QAction::triggered,[this](){
        auto f = QFontDialog::getFont(0,font());
        QApplication::setFont(f);
        setFont(f);

        QSettings s;
        s.beginGroup(BC::Key::BC);
        s.setValue(BC::Key::appFont,f);
        s.endGroup();
    });

    connect(ui->savePathAction,&QAction::triggered,[this](){
        if(p_batchManager && !p_batchManager->isComplete())
            return;
        BCSavePathDialog d(this);
        if(d.exec() == QDialog::Accepted)
        {
            SettingsStorage s;
            ui->exptSpinBox->setValue(s.get(BC::Key::exptNum,0));
        }
    });

    p_lh = new LogHandler(true,this);
    connect(this,&MainWindow::logMessage,p_lh,&LogHandler::logMessage);
    connect(p_lh,&LogHandler::sendLogMessage,ui->logTextEdit,&QTextEdit::append);
    connect(p_lh,&LogHandler::iconUpdate,this,&MainWindow::setLogIcon);
    connect(ui->mainTabWidget,&QTabWidget::currentChanged,[=](int i) {
        if(i == ui->mainTabWidget->indexOf(ui->logTab))
        {
            setLogIcon(LogHandler::Normal);
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

    connect(p_hwm,&HardwareManager::logMessage,p_lh,&LogHandler::logMessage);
    connect(p_hwm,&HardwareManager::statusMessage,ui->statusBar,&QStatusBar::showMessage);
    connect(p_hwm,&HardwareManager::allHardwareConnected,this,&MainWindow::hardwareInitialized);

    connect(p_hwm,&HardwareManager::clockFrequencyUpdate,ui->clockBox,&ClockDisplayBox::updateFrequency);


    for(auto it = d_hardware.cbegin(); it != d_hardware.cend(); ++it)
    {
        auto fullkey = it->first;
        auto l = fullkey.split("-");
        if(l.size() != 2)
            continue;

        auto key = l.at(0);
        auto index = l.at(1).toInt();

        auto act = ui->menuHardware->addAction(QString("%1: %2").arg(fullkey, p_hwm->getHwName(fullkey)));
        act->setObjectName(QString("Action")+key);



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
            connect(p_hwm,&HardwareManager::pGenRepRateUpdate,psb,&PulseStatusBox::updateRepRate);
            connect(p_hwm,&HardwareManager::pGenModeUpdate,psb,&PulseStatusBox::updatePGenMode);
            connect(p_hwm,&HardwareManager::pGenPulsingUpdate,psb,&PulseStatusBox::updatePGenEnabled);

            connect(act,&QAction::triggered,[this,psb,key]{
               if(isDialogOpen(key))
                   return;

               auto pcw = new PulseConfigWidget;
               auto pc = p_hwm->getPGenConfig();
               pcw->setFromConfig(pc);


               connect(p_hwm,&HardwareManager::pGenConfigUpdate,pcw,&PulseConfigWidget::setFromConfig);
               connect(p_hwm,&HardwareManager::pGenSettingUpdate,pcw,&PulseConfigWidget::newSetting);
               connect(p_hwm,&HardwareManager::pGenRepRateUpdate,pcw,&PulseConfigWidget::newRepRate);
               connect(p_hwm,&HardwareManager::pGenModeUpdate,pcw,&PulseConfigWidget::newSysMode);
               connect(p_hwm,&HardwareManager::pGenPulsingUpdate,pcw,&PulseConfigWidget::newPGenPulsing);
               connect(pcw,&PulseConfigWidget::changeSetting,p_hwm,&HardwareManager::setPGenSetting);
               connect(pcw,&PulseConfigWidget::changeRepRate,p_hwm,&HardwareManager::setPGenRepRate);
               connect(pcw,&PulseConfigWidget::changeSysMode,p_hwm,&HardwareManager::setPGenMode);
               connect(pcw,&PulseConfigWidget::changeSysPulsing,p_hwm,&HardwareManager::setPGenPulsingEnabled);

               createHWDialog(key,pcw);
            });

        }
        else if(key == BC::Key::TC::key)
        {
            auto tsb = new TemperatureStatusBox;
            tsb->setObjectName(key);
            ui->instrumentStatusLayout->insertWidget(3,tsb,0);
            connect(p_hwm,&HardwareManager::temperatureEnableUpdate,tsb,&TemperatureStatusBox::setChannelEnabled);
            connect(p_hwm,&HardwareManager::temperatureUpdate,tsb,&TemperatureStatusBox::setTemperature);
            connect(act,&QAction::triggered,[this,key,tsb](){
               if(isDialogOpen(key))
                   return;

               auto tcw = new TemperatureControlWidget;
               auto tc = p_hwm->getTemperatureControllerConfig();
               tcw->setFromConfig(tc);

               connect(p_hwm,&HardwareManager::temperatureEnableUpdate,tcw,&TemperatureControlWidget::setChannelEnabled);
               connect(tcw,&TemperatureControlWidget::channelEnableChanged,
                       p_hwm,&HardwareManager::setTemperatureChannelEnabled);
               connect(tcw,&TemperatureControlWidget::channelNameChanged
                       ,p_hwm,&HardwareManager::setTemperatureChannelName);
               connect(tcw,&TemperatureControlWidget::channelNameChanged,tsb,&TemperatureStatusBox::setChannelName);


               auto d = createHWDialog(key,tcw);
               connect(d,&QDialog::accepted,tsb,&TemperatureStatusBox::loadFromSettings);
            });
        }
#ifdef BC_LIF
        else if(key == BC::Key::LifLaser::key)
        {
            auto lsb = new LifLaserStatusBox;
            lsb->setObjectName(key);
            ui->instrumentStatusLayout->insertWidget(3,lsb,0);
            connect(p_hwm,&HardwareManager::lifLaserPosUpdate,lsb,&LifLaserStatusBox::setPosition);
            connect(p_hwm,&HardwareManager::lifLaserFlashlampUpdate,lsb,&LifLaserStatusBox::setFlashlampEnabled);
            connect(act,&QAction::triggered,[this,key,lsb](){
                if(isDialogOpen(key))
                    return;

               auto d = createHWDialog(key);
               connect(d,&QDialog::accepted,lsb,&LifLaserStatusBox::applySettings);
            });
        }
#endif
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
    connect(ui->pauseButton,&QToolButton::clicked,p_am,&AcquisitionManager::pause);
    connect(ui->resumeButton,&QToolButton::clicked,p_am,&AcquisitionManager::resume);
    connect(ui->abortButton,&QToolButton::clicked,p_am,&AcquisitionManager::abort);
    connect(p_am,&AcquisitionManager::backupComplete,ui->ftViewWidget,&FtmwViewWidget::updateBackups);
    connect(p_am,&AcquisitionManager::experimentComplete,ui->ftViewWidget,&FtmwViewWidget::experimentComplete);
    connect(p_am,&AcquisitionManager::experimentComplete,p_hwm,&HardwareManager::experimentComplete);

    QThread *amThread = new QThread(this);
    amThread->setObjectName("AcquisitionManagerThread");
    connect(amThread,&QThread::finished,p_am,&AcquisitionManager::deleteLater);
    p_am->moveToThread(amThread);
    d_threadObjectList.append(qMakePair(amThread,p_am));


    connect(p_hwm,&HardwareManager::experimentInitialized,this,&MainWindow::experimentInitialized);
    connect(p_hwm,&HardwareManager::ftmwScopeShotAcquired,p_am,&AcquisitionManager::processFtmwScopeShot);
    connect(p_am,&AcquisitionManager::newClockSettings,this,&MainWindow::clockPrompt);
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

    connect(ui->actionStart_Experiment,&QAction::triggered,this,&MainWindow::startExperiment);
    connect(ui->actionQuick_Experiment,&QAction::triggered,this,&MainWindow::quickStart);
    connect(ui->actionStart_Sequence,&QAction::triggered,this,&MainWindow::startSequence);
    connect(ui->pauseButton,&QToolButton::clicked,this,&MainWindow::pauseUi);
    connect(ui->resumeButton,&QToolButton::clicked,this,&MainWindow::resumeUi);
    connect(ui->actionCommunication,&QAction::triggered,this,&MainWindow::launchCommunicationDialog);
    connect(ui->actionRfConfig,&QAction::triggered,this,&MainWindow::launchRfConfigDialog);
    connect(ui->actionAutoscale_Aux,&QAction::triggered,ui->auxDataViewWidget,&AuxDataViewWidget::autoScaleAll);
    connect(ui->actionAutoscale_Rolling,&QAction::triggered,ui->rollingDataViewWidget,&RollingDataWidget::autoScaleAll);
    connect(ui->sleepButton,&QToolButton::toggled,this,&MainWindow::sleep);
    connect(ui->actionTest_All_Connections,&QAction::triggered,p_hwm,&HardwareManager::testAll);
    connect(ui->actionView_Experiment,&QAction::triggered,this,&MainWindow::viewExperiment);

#ifdef BC_LIF
    connect(ui->actionLifConfig,&QAction::triggered,this,&MainWindow::launchLifConfigDialog);
    connect(p_hwm,&HardwareManager::lifSettingsComplete,p_am,&AcquisitionManager::lifHardwareReady);
    connect(p_hwm,&HardwareManager::lifScopeShotAcquired,p_am,&AcquisitionManager::processLifScopeShot);
    connect(p_am,&AcquisitionManager::nextLifPoint,p_hwm,&HardwareManager::setLifParameters);
    connect(p_am,&AcquisitionManager::lifShotAcquired,ui->lifProgressBar,&QProgressBar::setValue);
    connect(p_am,&AcquisitionManager::lifPointUpdate,ui->lifDisplayWidget,&LifDisplayWidget::updatePoint);
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
    if(p_batchManager && !p_batchManager->isComplete())
        return;


    while(!d_openDialogs.empty())
    {
        //close any open HW dialogs
        auto d = d_openDialogs.extract(d_openDialogs.begin());
        d.mapped()->reject();
    }

    auto exp = std::make_shared<Experiment>();

    if(runExperimentWizard(exp.get()))
    {
        BatchManager *bm = new BatchSingle(exp);
        startBatch(bm);
    }
}

void MainWindow::quickStart()
{
    if(p_batchManager && !p_batchManager->isComplete())
        return;

    while(!d_openDialogs.empty())
    {
        //close any open HW dialogs
        auto d = d_openDialogs.extract(d_openDialogs.begin());
        d.mapped()->reject();
    }

    QuickExptDialog d(this);
    d.setHardware(d_hardware);
    int ret = d.exec();
    if(ret == QDialog::Rejected)
        return;
    else if(ret == QuickExptDialog::New)
    {
        startExperiment();
        return;
    }

    auto exp = std::make_shared<Experiment>(d.exptNumber(),"",true);
    if(ret == QuickExptDialog::Start)
    {
        configureOptionalHardware(exp.get(),&d);
        BatchManager *bm = new BatchSingle(exp);
        startBatch(bm);
        return;
    }
    else
    {
        if(runExperimentWizard(exp.get(),&d))
        {
            BatchManager *bm = new BatchSingle(exp);
            startBatch(bm);
        }
    }
}

void MainWindow::startSequence()
{
    if(p_batchManager && !p_batchManager->isComplete())
        return;


    while(!d_openDialogs.empty())
    {
        //close any open HW dialogs
        auto d = d_openDialogs.extract(d_openDialogs.begin());
        d.mapped()->reject();
    }

    BatchSequenceDialog d(this);
    d.setQuickExptEnabled(ui->exptSpinBox->value() > 0);
    int ret = d.exec();

    if(ret == QDialog::Rejected)
        return;

    std::shared_ptr<Experiment> exp = std::make_shared<Experiment>();

    if(ret == d.quickCode)
    {
        QuickExptDialog qed(this);
        qed.setHardware(d_hardware);
        int ret2 = qed.exec();
        if(ret2 == QDialog::Rejected)
            return;

        if(ret2 == QuickExptDialog::Start)
        {
            exp = std::make_shared<Experiment>(qed.exptNumber(),"",true);
            configureOptionalHardware(exp.get(),&qed);

            BatchSequence *bs = new BatchSequence(exp,d.numExperiments(),d.interval());
            startBatch(bs);
            return;
        }
        else if(ret2 == QuickExptDialog::Configure)
        {
            if(runExperimentWizard(exp.get(),&qed))
            {
                BatchSequence *bs = new BatchSequence(exp,d.numExperiments(),d.interval());
                startBatch(bs);
                return;
            }
        }

    }

    //if we reach this point, the experiment wizard needs to run
    if(runExperimentWizard(exp.get()))
    {
        BatchSequence *bs = new BatchSequence(exp,d.numExperiments(),d.interval());
        startBatch(bs);
    }

}

bool MainWindow::runExperimentWizard(Experiment *exp, QuickExptDialog *qed)
{
    configureOptionalHardware(exp,qed);

    ExperimentWizard wiz(exp,d_hardware,this);
    wiz.setValidationKeys(p_hwm->validationKeys());
    if(exp->ftmwEnabled())
        wiz.d_clocks = exp->ftmwConfig()->d_rfConfig.getClocks();
    else
    {
        QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks;
        QMetaObject::invokeMethod(p_hwm,&HardwareManager::getClocks,Qt::BlockingQueuedConnection,&clocks);
        wiz.d_clocks = clocks;
    }

#ifdef BC_LIF
    configureLifWidget(wiz.lifControlWidget());
#endif

    if(wiz.exec() != QDialog::Accepted)
        return false;

    return true;
}

void MainWindow::configureOptionalHardware(Experiment *exp, QuickExptDialog *qed)
{
    if(qed)
    {
        if((d_hardware.find(BC::Key::PGen::key) != d_hardware.end()) && qed->useCurrentSettings(BC::Key::PGen::key))
            exp->setPulseGenConfig(p_hwm->getPGenConfig());
        if((d_hardware.find(BC::Key::PGen::key) != d_hardware.end()) && qed->useCurrentSettings(BC::Key::Flow::flowController))
            exp->setFlowConfig(p_hwm->getFlowConfig());
        if((d_hardware.find(BC::Key::PController::key) != d_hardware.end()) && qed->useCurrentSettings(BC::Key::PController::key))
            exp->setPressureControllerConfig(p_hwm->getPressureControllerConfig());
    }
    else
    {
        if(d_hardware.find(BC::Key::PGen::key) != d_hardware.end())
            exp->setPulseGenConfig(p_hwm->getPGenConfig());
        if(d_hardware.find(BC::Key::Flow::flowController) != d_hardware.end())
            exp->setFlowConfig(p_hwm->getFlowConfig());
        if((d_hardware.find(BC::Key::PController::key) != d_hardware.end()))
            exp->setPressureControllerConfig(p_hwm->getPressureControllerConfig());
    }
}

void MainWindow::batchComplete(bool aborted)
{
    disconnect(p_am,&AcquisitionManager::auxData,ui->auxDataViewWidget,&AuxDataViewWidget::pointUpdated);
    disconnect(p_hwm,&HardwareManager::abortAcquisition,p_am,&AcquisitionManager::abort);

#ifdef BC_LIF
    ui->lifTab->setEnabled(true);
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

    if(d_state == Acquiring || d_state == Paused)
        configureUi(Idle);
}

void MainWindow::experimentInitialized(std::shared_ptr<Experiment> exp)
{   
    if(!p_batchManager || p_batchManager->isComplete())
        return;

    if(!exp->d_hardwareSuccess)
    {
        p_batchManager->experimentComplete();
        emit logMessage(QString("Hardware initialization unsuccessful."),LogHandler::Error);
        emit logMessage(exp->d_errorString,LogHandler::Error);
        configureUi(Idle);
        return;
    }

    if(!exp->initialize())
    {
        p_batchManager->experimentComplete();
        emit logMessage(QString("Could not initialize experiment."),LogHandler::Error);
        if(!exp->d_errorString.isEmpty())
            emit logMessage(exp->d_errorString,LogHandler::Error);
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
    ui->lifDisplayWidget->prepareForExperiment(*exp);
    if(exp->lifEnabled())
    {
        ui->lifTab->setEnabled(true);
        ui->lifProgressBar->setValue(0);
    }
    else
    {
        ui->lifTab->setEnabled(false);
        ui->lifProgressBar->setValue(1000);
    }
#endif

    if(!exp->isDummy())
        p_lh->beginExperimentLog(exp->d_number,exp->d_startLogMessage);
    else
        p_lh->logMessage(exp->d_startLogMessage,LogHandler::Highlight);

    //this is needed to reconfigure UI in case experiment is dummy
    configureUi(Acquiring);

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

void MainWindow::clockPrompt(QHash<RfConfig::ClockType, RfConfig::ClockFreq> c)
{
    auto up = c.value(RfConfig::UpLO);
    auto down = c.value(RfConfig::DownLO);

    bool upTunable = true;
    if(!up.hwKey.isEmpty())
    {
        SettingsStorage s(up.hwKey,SettingsStorage::Hardware);
        upTunable = s.get(BC::Key::Clock::tunable,true);
    }
    bool downTunable = true;
    if(!down.hwKey.isEmpty())
    {
        SettingsStorage s(down.hwKey,SettingsStorage::Hardware);
        downTunable = s.get(BC::Key::Clock::tunable,true);
    }

    if(!upTunable || !downTunable)
    {
        QMessageBox m;
        m.setWindowTitle(QString("Update LO Frequency"));
        m.setInformativeText(QString("Ensure your upconversion and/or downconversion LOs are set to the indicated frequencies. Press Ok (or hit enter) to proceed or Abort (escape) to terminate the acquisition."));

        QString displayString = QString("<table style=\"font-size:50pt;font-weight:bold\", cellpadding=\"20\">");
        if(!upTunable)
            displayString.append(QString("<tr><td>UpLO</td><td>%1</td><td>MHz</td></tr>").arg(QString::number(RfConfig::getRawFrequency(up),'f',6)));
        if(!downTunable)
            displayString.append(QString("<tr><td>DownLO</td><td>%1</td><td>MHz</td></tr>").arg(QString::number(RfConfig::getRawFrequency(down),'f',6)));
        displayString.append(QString("</table>"));
        m.setText(displayString);
        m.setTextFormat(Qt::RichText);
        m.setStandardButtons(QMessageBox::Ok|QMessageBox::Abort);

        auto ret = m.exec();
        if(ret == QMessageBox::Abort)
        {
            QMetaObject::invokeMethod(p_am,&AcquisitionManager::abort);
            return;
        }
    }

    QMetaObject::invokeMethod(p_hwm,[this,c](){p_hwm->setClocks(c);});

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
    auto it = d_openDialogs.find("RfConfig");
    if(it != d_openDialogs.end())
    {
        it->second->setWindowState(Qt::WindowActive);
        it->second->raise();
        it->second->show();
        return;
    }

    auto d = new QDialog;
    d->setWindowTitle("Rf Configuration");
    auto w = new RfConfigWidget(d);
    QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks;
    QMetaObject::invokeMethod(p_hwm,&HardwareManager::getClocks,Qt::BlockingQueuedConnection,&clocks);
    w->setClocks(clocks);

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
    connect(d,&QDialog::finished,d,&QDialog::deleteLater);
    connect(d,&QDialog::destroyed,[this](){
        auto it = d_openDialogs.find("RfConfig");
        if(it != d_openDialogs.end())
            d_openDialogs.erase(it);
    });

    d_openDialogs.insert({"RfConfig",d});
    d->show();

}

#ifdef BC_LIF
void MainWindow::launchLifConfigDialog()
{
    auto it = d_openDialogs.find("LifConfig");
    if(it != d_openDialogs.end())
    {
        it->second->setWindowState(Qt::WindowActive);
        it->second->raise();
        it->second->show();
        return;
    }

    auto d = new QDialog;
    d->setWindowTitle("LIF Configuration");

    auto w = new LifControlWidget(d);
    configureLifWidget(w);


//    auto lbl = new QLabel("TODO");
//    lbl->setWordWrap(true);

    auto vbl = new QVBoxLayout;
//    vbl->addWidget(lbl);
    vbl->addWidget(w);

    auto bb = new QDialogButtonBox(QDialogButtonBox::Close,d);
    connect(bb->button(QDialogButtonBox::Close),&QPushButton::clicked,d,&QDialog::reject);
    vbl->addWidget(bb);

    d->setLayout(vbl);
    connect(d,&QDialog::finished,p_hwm,&HardwareManager::stopLifConfigAcq);
    connect(d,&QDialog::finished,d,&QDialog::deleteLater);
    connect(d,&QDialog::destroyed,[this](){
        auto it = d_openDialogs.find("LifConfig");
        if(it != d_openDialogs.end())
            d_openDialogs.erase(it);
    });

    d_openDialogs.insert({"LifConfig",d});
    d->show();
}

void MainWindow::configureLifWidget(LifControlWidget *w)
{
    connect(w,&LifControlWidget::startSignal,p_hwm,&HardwareManager::startLifConfigAcq);
    connect(p_hwm,&HardwareManager::lifConfigAcqStarted,w,&LifControlWidget::acquisitionStarted);
    connect(w,&LifControlWidget::stopSignal,p_hwm,&HardwareManager::stopLifConfigAcq);
    connect(p_hwm,&HardwareManager::lifScopeShotAcquired,w,&LifControlWidget::newWaveform);
    connect(w,&LifControlWidget::changeLaserPosSignal,p_hwm,&HardwareManager::setLifLaserPos);
    connect(p_hwm,&HardwareManager::lifLaserPosUpdate,w,&LifControlWidget::setLaserPosition);
    connect(w,&LifControlWidget::changeLaserFlashlampSignal,p_hwm,&HardwareManager::setLifLaserFlashlampEnabled);
    connect(p_hwm,&HardwareManager::lifLaserFlashlampUpdate,w,&LifControlWidget::setFlashlamp);

    QMetaObject::invokeMethod(p_hwm,&HardwareManager::lifLaserPos,Qt::BlockingQueuedConnection);
    QMetaObject::invokeMethod(p_hwm,&HardwareManager::lifLaserFlashlampEnabled,Qt::BlockingQueuedConnection);

}
#endif

void MainWindow::setLogIcon(LogHandler::MessageCode c)
{
    if(ui->mainTabWidget->currentWidget() != ui->logTab)
    {
        switch(c) {
        case LogHandler::Warning:
            if(d_logIcon != LogHandler::Error)
            {
                ui->mainTabWidget->setTabIcon(ui->mainTabWidget->indexOf(ui->logTab),QIcon(QString(":/icons/warning.png")));
                d_logIcon = c;
            }
            break;
        case LogHandler::Error:
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
        d_logIcon = LogHandler::Normal;
        ui->mainTabWidget->setTabIcon(ui->mainTabWidget->indexOf(ui->logTab),QIcon());
    }
}

void MainWindow::sleep(bool s)
{
    if(d_state == Acquiring)
    {
        QMessageBox mb(this);
        mb.setWindowTitle(QString("Sleep when complete"));
        mb.setText(QString("Blackchirp will sleep when the current experiment (or batch) is complete."));
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
            QMessageBox::information(this,QString("Blackchirp Asleep"),QString("The instrument is asleep. Press the sleep button to re-activate it."),QMessageBox::Ok);
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
            QMessageBox::information(this,QString("Blackchirp Asleep"),QString("The instrument is asleep. Press the sleep button to re-activate it."),QMessageBox::Ok);
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
    if(p_batchManager && !p_batchManager->isComplete()
            && d_currentExptNum == lastCompletedExperiment)
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
    connect(out,&QDialog::finished,out,&QDialog::deleteLater);
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

    //start by disabling all actions; then enable as needed
    auto hwl = ui->menuHardware->actions();
    auto acq = ui->menuAcquisition->actions();

    for(auto act : hwl)
        act->setEnabled(false);
    for(auto act : acq)
        act->setEnabled(false);

    ui->abortButton->setEnabled(false);
    ui->pauseButton->setEnabled(false);
    ui->resumeButton->setEnabled(false);
    ui->sleepButton->setEnabled(false);
    ui->savePathAction->setEnabled(false);


    switch(s)
    {
    case Asleep:
        ui->sleepButton->setEnabled(true);
        ui->savePathAction->setEnabled(true);
        break;
    case Disconnected:
        ui->actionCommunication->setEnabled(true);
        ui->actionTest_All_Connections->setEnabled(true);
        ui->savePathAction->setEnabled(true);
        break;
    case Paused:
        ui->abortButton->setEnabled(true);
        ui->resumeButton->setEnabled(true);
        break;
    case Acquiring:
        ui->abortButton->setEnabled(true);
        ui->pauseButton->setEnabled(true);
        ui->sleepButton->setEnabled(true);
        if(p_batchManager && p_batchManager->currentExperiment()->isDummy())
        {
            for(auto act : hwl)
            {
                if(act == ui->actionRfConfig)
                    continue;
                if(act == ui->actionCommunication)
                    continue;
                if(act == ui->actionTest_All_Connections)
                    continue;
                act->setEnabled(true);
            }
        }
        break;
    case Idle:
    default:
        for(auto act : hwl)
            act->setEnabled(true);
        for(auto act : acq)
            act->setEnabled(true);
        ui->sleepButton->setEnabled(true);
        ui->savePathAction->setEnabled(true);
        break;
    }
}

void MainWindow::startBatch(BatchManager *bm)
{
    delete p_batchManager;

    connect(bm,&BatchManager::statusMessage,ui->statusBar,&QStatusBar::showMessage);
    connect(bm,&BatchManager::logMessage,p_lh,&LogHandler::logMessage);
    connect(bm,&BatchManager::beginExperiment,p_lh,&LogHandler::endExperimentLog);
    connect(bm,&BatchManager::beginExperiment,[this,bm](){p_hwm->initializeExperiment(bm->currentExperiment());});
    connect(p_am,&AcquisitionManager::experimentComplete,bm,&BatchManager::experimentComplete);
    connect(p_am,&AcquisitionManager::experimentComplete,ui->ftViewWidget,&FtmwViewWidget::experimentComplete);
    connect(ui->abortButton,&QToolButton::clicked,bm,&BatchManager::abort);
    connect(bm,&BatchManager::batchComplete,this,&MainWindow::batchComplete);
    connect(bm,&BatchManager::batchComplete,this,&MainWindow::checkSleep);
    connect(bm,&BatchManager::batchComplete,p_lh,&LogHandler::endExperimentLog);

#ifdef BC_LIF
    connect(p_am,&AcquisitionManager::experimentComplete,ui->lifDisplayWidget,&LifDisplayWidget::experimentComplete);
#endif

    connect(p_am,&AcquisitionManager::auxData,ui->auxDataViewWidget,&AuxDataViewWidget::pointUpdated,Qt::UniqueConnection);
    connect(p_hwm,&HardwareManager::abortAcquisition,p_am,&AcquisitionManager::abort,Qt::UniqueConnection);
    p_batchManager = bm;

    configureUi(Acquiring);

    QMetaObject::invokeMethod(p_hwm,[this](){ p_hwm->initializeExperiment(p_batchManager->currentExperiment());});

}

void MainWindow::closeEvent(QCloseEvent *ev)
{
    if(p_batchManager && !p_batchManager->isComplete())
        ev->ignore();
    else
    {      
        while(!d_openDialogs.empty())
        {
            //close any open HW dialogs
            auto d = d_openDialogs.extract(d_openDialogs.begin());
            d.mapped()->reject();
        }

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
