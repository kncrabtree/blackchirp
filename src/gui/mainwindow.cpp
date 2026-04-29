#include "mainwindow.h"
#include "mainwindow_ui.h"

#include <QThread>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QCloseEvent>
#include <QEvent>
#include <QFontMetrics>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QMessageBox>
#include <QCheckBox>
#include <QToolButton>
#include <QFileDialog>
#include <QDir>
#include <QDesktopServices>
#include <QMouseEvent>
#include <QScreen>
#include <QTimer>
#include <QUrl>
#include <QActionGroup>
#include <functional>

#include <gui/widget/digitizerconfigwidget.h>
#include <gui/widget/rfconfigwidget.h>
#include <gui/widget/pulseconfigwidget.h>
#include <gui/widget/gascontrolwidget.h>
#include <gui/widget/gasflowdisplaywidget.h>
#include <gui/widget/pulsestatusbox.h>
#include <gui/widget/temperaturestatusbox.h>
#include <gui/widget/temperaturecontrolwidget.h>
#include <gui/widget/led.h>
#include <gui/widget/experimentviewwidget.h>
#include <gui/widget/clockdisplaybox.h>
#include <gui/widget/gascontrolwidget.h>
#include <gui/widget/pressurestatusbox.h>
#include <gui/widget/pressurecontrolwidget.h>
#include <gui/style/themecolors.h>

#include <gui/dialog/aboutdialog.h>
#include <gui/dialog/applicationconfigdialog.h>
#include <gui/dialog/communicationdialog.h>
#include <gui/dialog/hwdialog.h>
#include <gui/dialog/quickexptdialog.h>
#include <gui/dialog/batchsequencedialog.h>
#include <gui/dialog/ftmwconfigdialog.h>
#include <gui/dialog/runtimehardwareconfigdialog.h>

// #include <gui/wizard/experimentwizard.h>
#include <gui/expsetup/experimentsetupdialog.h>

#include <data/loghandler.h>
#include <data/storage/blackchirpcsv.h>
#include <acquisition/acquisitionmanager.h>
#include <acquisition/batch/batchmanager.h>
#include <acquisition/batch/batchsingle.h>
#include <acquisition/batch/batchsequence.h>

#include <gui/lif/gui/lifdisplaywidget.h>
#include <gui/lif/gui/lifcontrolwidget.h>
#include <gui/lif/gui/liflaserstatusbox.h>
#include <hardware/core/liflaser/liflaser.h>
#include <data/storage/applicationconfigmanager.h>

#include <hardware/core/hardwaremanager.h>
#include <hardware/core/hardwareprofilemanager.h>
#include <hardware/core/runtimehardwareconfig.h>
#include <hardware/core/ftmwdigitizer/ftmwscope.h>
#include <data/loadout/loadoutmanager.h>
#include <data/settings/hardwarekeys.h>
#include <hardware/core/clock/fixedclock.h>
#include <gui/widget/pythonhardwarecontrolwidget.h>
#include <hardware/optional/tempcontroller/temperaturecontroller.h>
#include <hardware/optional/pressurecontroller/pressurecontroller.h>
#include <hardware/optional/flowcontroller/flowcontroller.h>
#include <hardware/optional/pulsegenerator/pulsegenerator.h>

#define _BC_STR(x) #x
#define BC_STRINGIFY(x) _BC_STR(x)

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    p_hwm = new HardwareManager();

    qRegisterMetaType<QwtPlot::Axis>("QwtPlot::Axis");

    ui->setupUi(this);
    
    // Apply theme-aware styling for SVG icons
    setupThemeAwareIconStyling();
    
    ui->rollingDurationBox->setValue(ui->rollingDataViewWidget->historyDuration());
    connect(ui->rollingDurationBox,&SpinBoxWidgetAction::valueChanged,
            ui->rollingDataViewWidget,&RollingDataWidget::setHistoryDuration);

    ui->auxGraphsBox->setValue(ui->auxDataViewWidget->numPlots());
    connect(ui->auxGraphsBox,&SpinBoxWidgetAction::valueChanged,
            ui->auxDataViewWidget,&AuxDataViewWidget::changeNumPlots);

    ui->rollingGraphsBox->setValue(ui->rollingDataViewWidget->numPlots());
    connect(ui->rollingGraphsBox,&SpinBoxWidgetAction::valueChanged,
            ui->rollingDataViewWidget,&RollingDataWidget::changeNumPlots);

    connect(ui->appConfigAction, &QAction::triggered, [this]() {
        if(p_batchManager && !p_batchManager->isComplete())
            return;
        ApplicationConfigDialog d(false, this);
        if(d.exec() == QDialog::Accepted)
        {
            SettingsStorage s;
            ui->exptValueLabel->setText(QString::number(s.get(BC::Key::exptNum, 0)));
            updateSavePathLabel();
        }
    });

    connect(ui->exptConfigButton, &QToolButton::clicked, ui->appConfigAction, &QAction::trigger);
    ui->savePathLabel->installEventFilter(this);

    p_lh = &LogHandler::instance();
    p_lh->setDebugLogging(ApplicationConfigManager::instance().isDebugLoggingEnabled());
    connect(&ApplicationConfigManager::instance(), &ApplicationConfigManager::debugLoggingChanged,
            p_lh, &LogHandler::setDebugLogging);
    connect(&ApplicationConfigManager::instance(), &ApplicationConfigManager::fontChanged,
            this, [this](QFont f) {
        QApplication::setFont(f);
        setFont(f);
    });
    connect(this,&MainWindow::logMessage,p_lh,&LogHandler::logMessage);
    connect(p_lh,&LogHandler::sendLogMessage,ui->logTextEdit,&QTextEdit::append);
    connect(p_lh,&LogHandler::iconUpdate,this,&MainWindow::setLogIcon);
    connect(ui->mainTabWidget,&QTabWidget::currentChanged,[this](int i) {
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
    connect(p_lh,&LogHandler::sendLogMessage,this,[this](){
        if(ui->mainTabWidget->currentWidget() != ui->logTab)
        {
            d_logCount++;
            ui->mainTabWidget->setTabText(ui->mainTabWidget->indexOf(ui->logTab),QString("Log (%1)").arg(d_logCount));
        }
    });

    connect(p_hwm,&HardwareManager::statusMessage,ui->statusBar,&QStatusBar::showMessage);
    connect(p_hwm,&HardwareManager::allHardwareConnected,this,&MainWindow::hardwareInitialized);

    // Add per-hardware connection tracking:
    connect(p_hwm, &HardwareManager::connectionResult, 
            this, &MainWindow::updateHardwareConnectionState);

    connect(p_hwm,&HardwareManager::clockFrequencyUpdate,ui->clockBox,&ClockDisplayBox::updateFrequency);
    connect(p_hwm,&HardwareManager::clockHardwareUpdate,ui->clockBox,&ClockDisplayBox::setClockHardware);
    connect(ui->clockBox,&ClockDisplayBox::configureRequested,this,&MainWindow::launchFtmwConfigDialog);
    connect(ui->clockBox,&ClockDisplayBox::clockHardwareRequested,this,[this](const QString &hwKey){
        auto it = d_hardwareUI.find(hwKey);
        if(it != d_hardwareUI.end() && it->second.menuAction)
            it->second.menuAction->trigger();
    });

    connect(p_hwm, &HardwareManager::profileDeleted, this, [this](const QString& hwKey) {
        auto it = d_openDialogs.find(hwKey);
        if (it != d_openDialogs.end()) {
            if (auto dlg = qobject_cast<HWDialog*>(it->second)) {
                dlg->discardControlWidget();
                dlg->reject();
            }
        }
    });

    // Build hardware UI dynamically - can now be called when hardware configuration changes
    buildHardwareUI();

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
    connect(ui->actionFtmwConfig,&QAction::triggered,this,&MainWindow::launchFtmwConfigDialog);
    connect(ui->actionAutoscale_Aux,&QAction::triggered,ui->auxDataViewWidget,&AuxDataViewWidget::autoScaleAll);
    connect(ui->actionAutoscale_Rolling,&QAction::triggered,ui->rollingDataViewWidget,&RollingDataWidget::autoScaleAll);
    connect(ui->sleepButton,&QToolButton::toggled,this,&MainWindow::sleep);
    connect(ui->actionTest_All_Connections,&QAction::triggered,p_hwm,&HardwareManager::testAll);
    connect(ui->viewExperimentAction,&QAction::triggered,this,&MainWindow::viewExperiment);

    if(ui->actionLifConfig)
        connect(ui->actionLifConfig,&QAction::triggered,this,&MainWindow::launchLifConfigDialog);
    connect(ui->actionRuntimeHardwareConfig,&QAction::triggered,this,&MainWindow::launchRuntimeHardwareConfigDialog);

    p_loadoutActionGroup = new QActionGroup(this);
    p_loadoutActionGroup->setExclusive(true);
    connect(ui->menuLoadout, &QMenu::triggered, this, &MainWindow::onLoadoutActionTriggered);
    auto &lm = LoadoutManager::instance();
    connect(&lm, &LoadoutManager::loadoutAdded, this, &MainWindow::rebuildLoadoutMenu);
    connect(&lm, &LoadoutManager::loadoutRemoved, this, &MainWindow::rebuildLoadoutMenu);
    connect(&lm, &LoadoutManager::loadoutChanged, this, &MainWindow::rebuildLoadoutMenu);
    connect(&lm, &LoadoutManager::currentLoadoutChanged, this, &MainWindow::rebuildLoadoutMenu);
    connect(&lm, &LoadoutManager::defaultLoadoutChanged, this, &MainWindow::rebuildLoadoutMenu);
    rebuildLoadoutMenu();

    p_ftmwPresetActionGroup = new QActionGroup(this);
    p_ftmwPresetActionGroup->setExclusive(true);
    connect(ui->menuFtmwPreset, &QMenu::triggered, this, &MainWindow::onFtmwPresetActionTriggered);
    connect(&lm, &LoadoutManager::currentLoadoutChanged, this, &MainWindow::rebuildFtmwPresetMenu);
    connect(&lm, &LoadoutManager::loadoutChanged, this, &MainWindow::rebuildFtmwPresetMenu);
    connect(&lm, &LoadoutManager::ftmwPresetAdded, this, &MainWindow::rebuildFtmwPresetMenu);
    connect(&lm, &LoadoutManager::ftmwPresetRemoved, this, &MainWindow::rebuildFtmwPresetMenu);
    connect(&lm, &LoadoutManager::ftmwPresetChanged, this, &MainWindow::rebuildFtmwPresetMenu);
    connect(&lm, &LoadoutManager::currentFtmwPresetChanged, this, &MainWindow::rebuildFtmwPresetMenu);
    rebuildFtmwPresetMenu();
    if(ApplicationConfigManager::instance().isLifEnabled())
    {
        connect(p_hwm,&HardwareManager::lifSettingsComplete,p_am,&AcquisitionManager::lifHardwareReady);
        connect(p_hwm,&HardwareManager::lifScopeShotAcquired,p_am,&AcquisitionManager::processLifScopeShot);
        connect(p_am,&AcquisitionManager::nextLifPoint,p_hwm,&HardwareManager::setLifParameters);
        connect(p_am,&AcquisitionManager::lifShotAcquired,ui->lifProgressBar,&QProgressBar::setValue);
        connect(p_am,&AcquisitionManager::lifPointUpdate,ui->lifDisplayWidget,&LifDisplayWidget::updatePoint);
    }

    SettingsStorage bc;
    ui->exptValueLabel->setText(QString::number(bc.get<int>(BC::Key::exptNum,0)));
    updateSavePathLabel();

    {
        using namespace Qt::Literals::StringLiterals;
        auto addUrl = [this](const QString &text, const char *url) {
            ui->helpMenu->addAction(text, this, [url]() {
                QDesktopServices::openUrl(QUrl(QLatin1StringView(url)));
            });
        };
        addUrl("&Documentation"_L1, "https://blackchirp.readthedocs.io/en/latest/index.html");
        addUrl("&GitHub Repository"_L1, "https://github.com/kncrabtree/blackchirp");
        addUrl("Di&scord Server"_L1, "https://discord.gg/88CkbAKUZY");
        ui->helpMenu->addSeparator();
        ui->helpMenu->addAction("&About Blackchirp"_L1, this, [this]() {
            using namespace Qt::Literals::StringLiterals;
            AboutDialog::AppInfo info;
            info.name = "Blackchirp"_L1;
            info.version = u"%1.%2.%3-%4"_s
                .arg(BC_MAJOR_VERSION).arg(BC_MINOR_VERSION)
                .arg(BC_PATCH_VERSION).arg(BC_STRINGIFY(BC_RELEASE_VERSION));
            info.build = QLatin1StringView(BC_BUILD_VERSION);
            info.description = "CP-FTMW spectroscopy data acquisition and visualization software."_L1;
            info.features = {
                {"CUDA"_L1,
#ifdef BC_CUDA
                 "Enabled"_L1
#else
                 "Disabled"_L1
#endif
                },
                {"LIF module"_L1, ApplicationConfigManager::instance().isLifEnabled()
                     ? QString("Enabled"_L1) : QString("Disabled"_L1)},
            };
            AboutDialog dlg(info, this);
            dlg.exec();
        });
        ui->helpMenu->addAction("About &Qt"_L1, qApp, &QApplication::aboutQt);
    }

    // Defer UI configuration until after the widget is fully rendered
    // This prevents LIF widgets from briefly appearing on wrong tabs during initialization
    QTimer::singleShot(0, this, [this]() { configureUi(Idle); });
}

void MainWindow::buildHardwareUI()
{
    auto currentHardware = RuntimeHardwareConfig::constInstance().getCurrentHardware();
    for(auto it = currentHardware.cbegin(); it != currentHardware.cend(); ++it)
    {
        auto key = it->first;
        auto implName = it->second;
        auto ki = BC::Key::parseKey(key);

        auto hwType = ki.first;

        HardwareUIElements elements;

        auto act = ui->menuHardware->addAction(key);
        act->setObjectName(Ui::actionStr+key);
        elements.menuAction = act;

        if(hwType == QString(FlowController::staticMetaObject.className()))
        {
            auto w = new GasFlowDisplayBox(key);
            w->setObjectName(key+Ui::sbStr);
            ui->hwStatusLayout->addWidget(w);
            elements.statusWidget = w;
            elements.connections.append(connect(w,&HardwareStatusBox::configureRequested,act,&QAction::trigger));

            elements.connections.append(connect(p_hwm,&HardwareManager::flowUpdate,w,&GasFlowDisplayBox::updateFlow));
            elements.connections.append(connect(p_hwm,&HardwareManager::flowSetpointUpdate,w,&GasFlowDisplayBox::updateFlowSetpoint));
            elements.connections.append(connect(p_hwm,&HardwareManager::gasPressureUpdate,w,&GasFlowDisplayBox::updatePressure));
            elements.connections.append(connect(p_hwm,&HardwareManager::gasPressureControlMode,w,&GasFlowDisplayBox::updatePressureControl));

            elements.connections.append(connect(act,&QAction::triggered,[this,w,key,implName]{

                if(isDialogOpen(key))
                    return;

                auto fc = p_hwm->getFlowConfig(key);
                auto gcw = new GasControlWidget(fc);
                connect(p_hwm,&HardwareManager::flowSetpointUpdate,gcw,&GasControlWidget::updateGasSetpoint);
                connect(p_hwm,&HardwareManager::gasPressureSetpointUpdate,gcw,&GasControlWidget::updatePressureSetpoint);
                connect(p_hwm,&HardwareManager::gasPressureControlMode,gcw,&GasControlWidget::updatePressureControl);
                connect(gcw,&GasControlWidget::pressureControlUpdate,p_hwm,&HardwareManager::setGasPressureControlMode);
                connect(gcw,&GasControlWidget::pressureSetpointUpdate,p_hwm,&HardwareManager::setGasPressureSetpoint);
                connect(gcw,&GasControlWidget::gasSetpointUpdate,p_hwm,&HardwareManager::setFlowSetpoint);
                connect(gcw,&GasControlWidget::nameUpdate,w,&GasFlowDisplayBox::updateFlowName);
                connect(gcw,&GasControlWidget::nameUpdate,p_hwm,&HardwareManager::setFlowChannelName);

                QWidget *cw = implName.contains(QStringLiteral("Python")) ? wrapWithPythonWidget(key, gcw) : gcw;
                auto d = createHWDialog(key,cw);
                connect(d,&QDialog::accepted,w,&GasFlowDisplayBox::rebuild);

            }));
        }
        else if(hwType == QString(PressureController::staticMetaObject.className()))
        {
            auto psb = new PressureStatusBox(key);
            psb->setObjectName(key);
            ui->hwStatusLayout->addWidget(psb);
            elements.statusWidget = psb;
            elements.connections.append(connect(psb,&HardwareStatusBox::configureRequested,act,&QAction::trigger));

            elements.connections.append(connect(p_hwm,&HardwareManager::pressureUpdate,psb,&PressureStatusBox::pressureUpdate));
            elements.connections.append(connect(p_hwm,&HardwareManager::pressureControlMode,psb,&PressureStatusBox::pressureControlUpdate));

            elements.connections.append(connect(act,&QAction::triggered,[this,psb,key,implName](){

                if(isDialogOpen(key))
                    return;

                auto pc = p_hwm->getPressureControllerConfig(key);
                auto pcw = new PressureControlWidget(pc);
                connect(p_hwm,&HardwareManager::pressureSetpointUpdate,pcw,&PressureControlWidget::pressureSetpointUpdate);
                connect(p_hwm,&HardwareManager::pressureControlMode,pcw,&PressureControlWidget::pressureControlModeUpdate);
                connect(pcw,&PressureControlWidget::setpointChanged,p_hwm,&HardwareManager::setPressureSetpoint);
                connect(pcw,&PressureControlWidget::pressureControlModeChanged,p_hwm,&HardwareManager::setPressureControlMode);
                connect(pcw,&PressureControlWidget::valveOpen,p_hwm,&HardwareManager::openGateValve);
                connect(pcw,&PressureControlWidget::valveClose,p_hwm,&HardwareManager::closeGateValve);

                QWidget *cw = implName.contains(QStringLiteral("Python")) ? wrapWithPythonWidget(key, pcw) : pcw;
                auto d = createHWDialog(key,cw);
                connect(d,&QDialog::accepted,psb,&PressureStatusBox::updateFromSettings);
            }));
        }
        else if(hwType == QString(PulseGenerator::staticMetaObject.className()))
        {
            auto psb = new PulseStatusBox(key);
            psb->setObjectName(key+Ui::sbStr);
            ui->hwStatusLayout->addWidget(psb);;
            elements.statusWidget = psb;
            elements.connections.append(connect(psb,&HardwareStatusBox::configureRequested,act,&QAction::trigger));

            elements.connections.append(connect(p_hwm,&HardwareManager::pGenConfigUpdate,psb,&PulseStatusBox::updatePulseLeds));
            elements.connections.append(connect(p_hwm,&HardwareManager::pGenSettingUpdate,psb,&PulseStatusBox::updatePulseSetting));

            elements.connections.append(connect(act,&QAction::triggered,[this,psb,key,implName]{
               if(isDialogOpen(key))
                   return;

               auto pc = p_hwm->getPGenConfig(key);
               auto pcw = new PulseConfigWidget(pc);


               connect(p_hwm,&HardwareManager::pGenConfigUpdate,pcw,&PulseConfigWidget::setFromConfig);
               connect(p_hwm,&HardwareManager::pGenSettingUpdate,pcw,&PulseConfigWidget::newSetting);
               connect(pcw,&PulseConfigWidget::changeSetting,p_hwm,&HardwareManager::setPGenSetting);
               QWidget *cw = implName.contains(QStringLiteral("Python")) ? wrapWithPythonWidget(key, pcw) : pcw;
               auto d = createHWDialog(key,cw);
               connect(d, &QDialog::accepted, psb, &PulseStatusBox::rebuild);
            }));

        }
        else if(hwType == QString(TemperatureController::staticMetaObject.className()))
        {
            auto tsb = new TemperatureStatusBox(key);
            tsb->setObjectName(key+Ui::sbStr);
            ui->hwStatusLayout->addWidget(tsb);
            elements.statusWidget = tsb;
            elements.connections.append(connect(tsb,&HardwareStatusBox::configureRequested,act,&QAction::trigger));

            elements.connections.append(connect(p_hwm,&HardwareManager::temperatureEnableUpdate,tsb,&TemperatureStatusBox::setChannelEnabled));
            elements.connections.append(connect(p_hwm,&HardwareManager::temperatureUpdate,tsb,&TemperatureStatusBox::setTemperature));
            elements.connections.append(connect(act,&QAction::triggered,[this,key,tsb,implName](){
               if(isDialogOpen(key))
                   return;

               auto tc = p_hwm->getTemperatureControllerConfig(key);
               auto tcw = new TemperatureControlWidget(tc);

               connect(p_hwm,&HardwareManager::temperatureEnableUpdate,tcw,&TemperatureControlWidget::setChannelEnabled);
               connect(tcw,&TemperatureControlWidget::channelEnableChanged,
                       p_hwm,&HardwareManager::setTemperatureChannelEnabled);
               connect(tcw,&TemperatureControlWidget::channelNameChanged
                       ,p_hwm,&HardwareManager::setTemperatureChannelName);
               connect(tcw,&TemperatureControlWidget::channelNameChanged,tsb,&TemperatureStatusBox::setChannelName);

               QWidget *cw = implName.contains(QStringLiteral("Python")) ? wrapWithPythonWidget(key, tcw) : tcw;
               auto d = createHWDialog(key,cw);
               connect(d,&QDialog::accepted,tsb,&TemperatureStatusBox::loadFromSettings);
            }));
        }
        else if(hwType == QString(LifLaser::staticMetaObject.className()))
        {
            auto lsb = new LifLaserStatusBox(key);
            lsb->setObjectName(key+Ui::sbStr);
            ui->hwStatusLayout->addWidget(lsb);
            elements.statusWidget = lsb;
            elements.connections.append(connect(lsb,&HardwareStatusBox::configureRequested,act,&QAction::trigger));

            elements.connections.append(connect(p_hwm,&HardwareManager::lifLaserPosUpdate,lsb,&LifLaserStatusBox::setPosition));
            elements.connections.append(connect(p_hwm,&HardwareManager::lifLaserFlashlampUpdate,lsb,&LifLaserStatusBox::setFlashlampEnabled));
            elements.connections.append(connect(act,&QAction::triggered,[this,key,lsb,implName](){
                if(isDialogOpen(key))
                    return;

               QWidget *cw = implName.contains(QStringLiteral("Python")) ? wrapWithPythonWidget(key, nullptr) : nullptr;
               auto d = createHWDialog(key, cw);
               connect(d,&QDialog::accepted,lsb,&LifLaserStatusBox::applySettings);
            }));
            
        }
        else
        {
            elements.statusWidget = nullptr; // No status widget for generic hardware
            elements.connections.append(connect(act,&QAction::triggered,[this,key,implName](){
                if(isDialogOpen(key))
                    return;

                QWidget *cw = implName.contains(QStringLiteral("Python")) ? wrapWithPythonWidget(key, nullptr) : nullptr;
                createHWDialog(key, cw);
            }));
        }

        d_hardwareUI[key] = elements;
        d_hardwareConnectionState[key] = false; // Initialize as disconnected
    }

    ui->hwStatusLayout->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Minimum,QSizePolicy::MinimumExpanding));
}

void MainWindow::clearHardwareUI()
{
    for(auto& [hwKey, elements] : d_hardwareUI) {
        // Disconnect all signals
        for(const auto& connection : elements.connections) {
            disconnect(connection);
        }
        
        // Remove and delete widgets
        if(elements.statusWidget) {
            ui->hwStatusLayout->removeWidget(elements.statusWidget);
            delete elements.statusWidget;
        }
        
        // Remove menu action
        if(elements.menuAction) {
            ui->menuHardware->removeAction(elements.menuAction);
            delete elements.menuAction;
        }
    }
    
    d_hardwareUI.clear();
    d_hardwareConnectionState.clear();
    
    // Remove the spacer item too
    QLayoutItem* spacer = ui->hwStatusLayout->takeAt(ui->hwStatusLayout->count()-1);
    delete spacer;
}

void MainWindow::updateHardwareConnectionState(const QString& hwKey, bool connected)
{
    d_hardwareConnectionState[hwKey] = connected;
    
    // Update individual UI element state
    if(d_hardwareUI.contains(hwKey)) {
        auto& elements = d_hardwareUI[hwKey];
        if(elements.menuAction)
            elements.menuAction->setEnabled(connected);
        if(elements.statusWidget)
            elements.statusWidget->setEnabled(connected);
        // Could add visual feedback (grayed out, different styling, etc.)
    }
    
    // Update overall UI state
    configureUiForHardwareState();
}

void MainWindow::configureUiForHardwareState()
{
    // Update overall UI state based on current hardware connection states
    // For now, trigger the standard UI configuration update
    configureUi(d_state);
}

bool MainWindow::isCriticalHardwareConnected() const
{
    // Ask the HardwareManager - it has the authoritative information about
    // hardware criticality and connection status
    return p_hwm ? p_hwm->allCriticalHardwareConnected() : false;
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

    auto exp = createExperiment();
    QMetaObject::invokeMethod(p_hwm,[this,exp]{
        p_hwm->storeAllOptHw(exp.get());
    },Qt::BlockingQueuedConnection);


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
    d.setQuickExptEnabled(ui->exptValueLabel->text().toInt() > 0);
    int ret = d.exec();

    if(ret == QDialog::Rejected)
        return;

    std::shared_ptr<Experiment> exp = createExperiment();

    if(ret == d.quickCode)
    {
        QuickExptDialog qed(this);
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
    QMetaObject::invokeMethod(p_hwm,[this,exp](){
        p_hwm->storeAllOptHw(exp.get());
    },Qt::BlockingQueuedConnection);
    if(runExperimentWizard(exp.get()))
    {
        BatchSequence *bs = new BatchSequence(exp,d.numExperiments(),d.interval());
        startBatch(bs);
    }

}

bool MainWindow::runExperimentWizard(Experiment *exp, QuickExptDialog *qed)
{
    configureOptionalHardware(exp,qed);

    QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks;
    if(exp->ftmwEnabled())
        clocks = exp->ftmwConfig()->d_rfConfig.getClocks();
    else
        QMetaObject::invokeMethod(p_hwm,&HardwareManager::getClocks,Qt::BlockingQueuedConnection,&clocks);

    ExperimentSetupDialog d(exp,clocks,p_hwm->validationKeys(),this);
    connectRfConfigWidget(d.rfConfigWidget());

    if(ApplicationConfigManager::instance().isLifEnabled()) {
        configureLifWidget(d.lifControlWidget());
    }

    if(d.exec() != QDialog::Accepted)
        return false;

    return true;
}

void MainWindow::configureOptionalHardware(Experiment *exp, QuickExptDialog *qed)
{
    if(qed)
    {
        auto l = qed->getOptHwSettings();
        QMetaObject::invokeMethod(p_hwm,[this,exp,l]{
           p_hwm->storeAllOptHw(exp,l);
        });
    }
}

void MainWindow::batchComplete(bool aborted)
{
    disconnect(p_am,&AcquisitionManager::auxData,ui->auxDataViewWidget,&AuxDataViewWidget::pointUpdated);
    disconnect(p_hwm,&HardwareManager::abortAcquisition,p_am,&AcquisitionManager::abort);

    if(ui->lifTab)
        ui->lifTab->setEnabled(true);

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
        bcError("Hardware initialization unsuccessful."_L1);
        bcError(exp->d_errorString);
        configureUi(Idle);
        return;
    }

    if(!exp->initialize())
    {
        p_batchManager->experimentComplete();
        bcError("Could not initialize experiment."_L1);
        if(!exp->d_errorString.isEmpty())
            bcError(exp->d_errorString);
        configureUi(Idle);
        return;
    }

    if(exp->d_number > 0)
        ui->exptValueLabel->setText(QString::number(exp->d_number));

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

    if(ui->lifDisplayWidget)
    {
        ui->lifDisplayWidget->prepareForExperiment(*exp);
        if(exp->lifEnabled())
        {
            if(ui->lifTab)
                ui->lifTab->setEnabled(true);
            if(ui->lifProgressBar)
                ui->lifProgressBar->setValue(0);
        }
        else
        {
            if(ui->lifTab)
                ui->lifTab->setEnabled(false);
            if(ui->lifProgressBar)
                ui->lifProgressBar->setValue(1000);
        }
    }

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

    if(!d_initialHardwareTestComplete)
    {
        d_initialHardwareTestComplete = true;
        if(!success)
            launchCommunicationDialog(true);
    }
}

void MainWindow::clockPrompt(QHash<RfConfig::ClockType, RfConfig::ClockFreq> c)
{
    auto up = c.value(RfConfig::UpLO);
    auto down = c.value(RfConfig::DownLO);
    auto dr = c.value(RfConfig::DRClock);


    bool upManual = false;
    if(!up.hwKey.isEmpty())
    {
        SettingsStorage s(up.hwKey,SettingsStorage::Hardware);
        upManual = s.get(BC::Key::Clock::manualTune,false);
    }
    bool downManual = false;
    if(!down.hwKey.isEmpty())
    {
        SettingsStorage s(down.hwKey,SettingsStorage::Hardware);
        downManual  = s.get(BC::Key::Clock::manualTune,false);
    }
    bool drManual = false;
    if(!dr.hwKey.isEmpty())
    {
        SettingsStorage s(dr.hwKey,SettingsStorage::Hardware);
        drManual  = s.get(BC::Key::Clock::manualTune,false);
    }

    if(upManual || downManual || drManual)
    {
        QMessageBox m;
        m.setWindowTitle(QString("Update LO Frequency"));
        m.setInformativeText(QString("Ensure LOs are set to the indicated frequencies. Press Ok (or hit enter) to proceed or Abort (escape) to terminate the acquisition."));

        QString displayString = QString("<table style=\"font-size:50pt;font-weight:bold\", cellpadding=\"20\">");
        if(upManual)
            displayString.append(QString("<tr><td>UpLO</td><td>%1</td><td>MHz</td></tr>").arg(QString::number(RfConfig::getRawFrequency(up),'f',6)));
        if(downManual)
            displayString.append(QString("<tr><td>DownLO</td><td>%1</td><td>MHz</td></tr>").arg(QString::number(RfConfig::getRawFrequency(down),'f',6)));
        if(drManual)
            displayString.append(QString("<tr><td>DR</td><td>%1</td><td>MHz</td></tr>").arg(QString::number(RfConfig::getRawFrequency(dr),'f',6)));
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

    d.exec();
}

void MainWindow::launchFtmwConfigDialog()
{
    auto it = d_openDialogs.find(BC::Key::Ftmw::ftmwDialogKey);
    if(it != d_openDialogs.end())
    {
        it->second->setWindowState(Qt::WindowActive);
        it->second->raise();
        it->second->show();
        return;
    }

    QString awgHwKey, digiHwKey;
    const auto hardware = RuntimeHardwareConfig::constInstance().getCurrentHardware();
    for(const auto &[k, v] : hardware)
    {
        auto [type, label] = BC::Key::parseKey(k);
        if(type == BC::Key::AWG::key)
            awgHwKey = k;
        else if(type == RuntimeHardwareConfig::hardwareTypeOf<FtmwScope>())
            digiHwKey = k;
    }

    QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks;
    QMetaObject::invokeMethod(p_hwm, &HardwareManager::getClocks, Qt::BlockingQueuedConnection, &clocks);

    auto d = new FtmwConfigDialog(awgHwKey, digiHwKey, clocks, this);

    connect(d, &FtmwConfigDialog::applyClocks, [this](QHash<RfConfig::ClockType, RfConfig::ClockFreq> c){
        QMetaObject::invokeMethod(p_hwm, [this, c](){ p_hwm->configureClocks(c); });
    });

    connect(d, &QDialog::accepted, [this](){
        auto activeName = LoadoutManager::instance().currentLoadoutName();
        auto preset = LoadoutManager::instance().currentFtmwPreset(activeName);
        if(!preset.has_value())
            return;
        const auto clocks = preset->rfConfig.clocks;
        QMetaObject::invokeMethod(p_hwm, [this, clocks](){ p_hwm->configureClocks(clocks); }, Qt::QueuedConnection);
    });

    connect(d, &QDialog::finished, d, &QDialog::deleteLater);
    connect(d, &QDialog::destroyed, [this](){
        auto it = d_openDialogs.find(BC::Key::Ftmw::ftmwDialogKey);
        if(it != d_openDialogs.end())
            d_openDialogs.erase(it);
    });

    d_openDialogs.insert({BC::Key::Ftmw::ftmwDialogKey, d});
    d->show();
}

void MainWindow::rebuildLoadoutMenu()
{
    for(auto *act : p_loadoutActionGroup->actions())
        p_loadoutActionGroup->removeAction(act);
    ui->menuLoadout->clear();
    const auto &lm = LoadoutManager::instance();
    const QString current = lm.currentLoadoutName();

    for(const auto &name : lm.loadoutNames())
    {
        auto *act = ui->menuLoadout->addAction(name);
        act->setCheckable(true);
        act->setChecked(name == current);
        act->setData(name);
        p_loadoutActionGroup->addAction(act);
    }
}

void MainWindow::onLoadoutActionTriggered(QAction *act)
{
    using namespace Qt::StringLiterals;
    const QString target = act->data().toString();
    const QString current = LoadoutManager::instance().currentLoadoutName();

    if(target == current)
    {
        act->setChecked(true);
        return;
    }

    auto result = QMessageBox::question(this, u"Switch Loadout"_s,
        u"Switch to loadout \"%1\"? This will reconfigure all hardware."_s.arg(target),
        QMessageBox::Yes | QMessageBox::Cancel);

    if(result != QMessageBox::Yes)
    {
        for(auto *a : p_loadoutActionGroup->actions())
        {
            if(a->data().toString() == current)
            {
                a->setChecked(true);
                break;
            }
        }
        return;
    }

    auto loadout = LoadoutManager::instance().getLoadout(target);
    if(!loadout)
        return;

    const auto map = loadout->hardwareMap;
    QMetaObject::invokeMethod(p_hwm, [this, map]{ p_hwm->applyHardwareMap(map); },
                              Qt::BlockingQueuedConnection);
    clearHardwareUI();
    buildHardwareUI();

    auto &pm = HardwareProfileManager::instance();
    pm.saveProfiles();
    QMetaObject::invokeMethod(p_hwm, &HardwareManager::syncWithRuntimeConfig, Qt::QueuedConnection);

    auto preset = LoadoutManager::instance().currentFtmwPreset(target);
    if(preset.has_value())
    {
        auto clocks = preset->rfConfig.clocks;
        QMetaObject::invokeMethod(p_hwm, [this, clocks](){
            p_hwm->configureClocks(clocks);
        }, Qt::QueuedConnection);
    }

    LoadoutManager::instance().setCurrentLoadoutName(target);
}

void MainWindow::rebuildFtmwPresetMenu()
{
    for(auto *act : p_ftmwPresetActionGroup->actions())
        p_ftmwPresetActionGroup->removeAction(act);
    ui->menuFtmwPreset->clear();

    const auto &lm = LoadoutManager::instance();
    const QString activeLoadout = lm.currentLoadoutName();
    const QStringList names = lm.ftmwPresetNames(activeLoadout, false);
    const QString current = lm.currentFtmwPresetName(activeLoadout);

    for(const auto &name : names)
    {
        auto *act = ui->menuFtmwPreset->addAction(name);
        act->setCheckable(true);
        act->setChecked(name == current);
        act->setData(name);
        p_ftmwPresetActionGroup->addAction(act);
    }

    ui->menuFtmwPreset->menuAction()->setEnabled(!names.isEmpty());
}

void MainWindow::onFtmwPresetActionTriggered(QAction *act)
{
    using namespace Qt::StringLiterals;
    const QString name = act->data().toString();
    const auto &lm = LoadoutManager::instance();
    const QString activeLoadout = lm.currentLoadoutName();
    const QString current = lm.currentFtmwPresetName(activeLoadout);

    if(name == current)
    {
        act->setChecked(true);
        return;
    }

    auto result = QMessageBox::question(this, u"Switch FTMW Preset"_s,
        u"Switch to FTMW preset \"%1\" of loadout \"%2\"? This will push new clock settings."_s
            .arg(name, activeLoadout),
        QMessageBox::Yes | QMessageBox::Cancel);

    if(result != QMessageBox::Yes)
    {
        for(auto *a : p_ftmwPresetActionGroup->actions())
        {
            if(a->data().toString() == current)
            {
                a->setChecked(true);
                break;
            }
        }
        return;
    }

    auto preset = lm.getFtmwPreset(activeLoadout, name);
    if(!preset.has_value())
        return;

    LoadoutManager::instance().setCurrentFtmwPresetName(activeLoadout, name);

    auto clocks = preset->rfConfig.clocks;
    QMetaObject::invokeMethod(p_hwm, [this, clocks](){
        p_hwm->configureClocks(clocks);
    }, Qt::BlockingQueuedConnection);
}

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

    // Check if LIF hardware is available before creating dialog
    auto& runtimeConfig = RuntimeHardwareConfig::constInstance();
    auto activeLabels = runtimeConfig.getActiveLabels<LifScope>();
    auto activeLabels2 = runtimeConfig.getActiveLabels<LifLaser>();
    if (activeLabels.isEmpty() || activeLabels2.isEmpty()) {
        QMessageBox::warning(this, "LIF Configuration", "Please configure LIF hardware before opening this dialog.");
        return;
    }

    auto d = new QDialog;
    d->setWindowTitle("LIF Configuration");

    // Create LifControlWidget with hardware keys
    auto scopeKeys = runtimeConfig.getActiveKeys<LifScope>();
    auto laserKeys = runtimeConfig.getActiveKeys<LifLaser>();

    auto w = new LifControlWidget(scopeKeys.first(), laserKeys.first(), d);
    configureLifWidget(w);

    auto vbl = new QVBoxLayout;
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

void MainWindow::launchRuntimeHardwareConfigDialog()
{
    auto it = d_openDialogs.find("RuntimeHardwareConfig");
    if(it != d_openDialogs.end())
    {
        it->second->setWindowState(Qt::WindowActive);
        it->second->raise();
        it->second->show();
        return;
    }

    // Capture current hardware configuration before dialog opens
    const auto& config = RuntimeHardwareConfig::constInstance();
    auto initialHardware = config.getCurrentHardware();

    auto d = new RuntimeHardwareConfigDialog(this);
    
    // Phase 3.4.3 - Dynamic UI integration with runtime hardware configuration
    // Check if configuration changed when dialog closes (regardless of accept/reject)
    // and rebuild UI before hardware synchronization
    connect(d, &QDialog::finished, [this, initialHardware]() {
        const auto& config = RuntimeHardwareConfig::constInstance();
        auto currentHardware = config.getCurrentHardware();
        
        // Check if hardware configuration actually changed
        if (initialHardware != currentHardware) {
            // UI must be rebuilt BEFORE hardware synchronization so that
            // UI elements exist to receive connectionResult signals during testing
            clearHardwareUI();
            buildHardwareUI();
        }
        
        // Trigger hardware synchronization - UI elements ready to receive signals
        QMetaObject::invokeMethod(p_hwm, &HardwareManager::syncWithRuntimeConfig, Qt::QueuedConnection);
    });
    
    connect(d, &QDialog::finished, [this](int result) {
        if (result != QDialog::Accepted)
            return;
        auto activeName = LoadoutManager::instance().currentLoadoutName();
        auto preset = LoadoutManager::instance().currentFtmwPreset(activeName);
        if (!preset.has_value())
            return;
        const auto clocks = preset->rfConfig.clocks;
        QMetaObject::invokeMethod(p_hwm, [this, clocks]() {
            p_hwm->configureClocks(clocks);
        }, Qt::QueuedConnection);
    });

    connect(d, &QDialog::finished, d, &QDialog::deleteLater);
    connect(d, &QDialog::destroyed, [this](){
        auto it = d_openDialogs.find("RuntimeHardwareConfig");
        if(it != d_openDialogs.end())
            d_openDialogs.erase(it);
    });
    
    d_openDialogs.insert({"RuntimeHardwareConfig", d});
    d->show();
}

void MainWindow::connectRfConfigWidget(RfConfigWidget *w)
{
    if(w)
    {
        connect(w,&RfConfigWidget::applyClocks,[this](QHash<RfConfig::ClockType, RfConfig::ClockFreq> c){
            QMetaObject::invokeMethod(p_hwm,[c,this](){ p_hwm->configureClocks(c); });
        });
    }
}

void MainWindow::configureLifWidget(LifControlWidget *w)
{
    if(w)
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

}

void MainWindow::setLogIcon(LogHandler::MessageCode c)
{
    if(ui->mainTabWidget->currentWidget() != ui->logTab)
    {
        switch(c) {
        case LogHandler::Warning:
            if(d_logIcon != LogHandler::Error)
            {
                ui->mainTabWidget->setTabIcon(ui->mainTabWidget->indexOf(ui->logTab), ThemeColors::createThemedIcon(":/icons/exclamation-triangle.svg", ThemeColors::StatusWarning, this));
                d_logIcon = c;
            }
            break;
        case LogHandler::Error:
            ui->mainTabWidget->setTabIcon(ui->mainTabWidget->indexOf(ui->logTab), ThemeColors::createThemedIcon(":/icons/exclamation-triangle.svg", ThemeColors::StatusError, this));
            d_logIcon = c;
            break;
        default:
            d_logIcon = c;
            ui->mainTabWidget->setTabIcon(ui->mainTabWidget->indexOf(ui->logTab), ThemeColors::createThemedIcon(":/icons/document-text.svg", ThemeColors::IconSecondary, this));
            break;
        }
    }
    else
    {
        d_logIcon = LogHandler::Normal;
        ui->mainTabWidget->setTabIcon(ui->mainTabWidget->indexOf(ui->logTab), ThemeColors::createThemedIcon(":/icons/document-text.svg", ThemeColors::IconSecondary, this));
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
    browseButton->setIcon(ThemeColors::createThemedIcon(":/icons/document-magnifying-glass.svg", ThemeColors::IconSecondary, this));

    connect(browseButton,&QToolButton::clicked,this,[this,pathEdit](){
        QString path = QFileDialog::getExistingDirectory(this,QString("Select experiment directory"),QString("~"));
        if(!path.isEmpty())
            pathEdit->setText(path);
    });

    hl->addWidget(pathEdit,1);
    hl->addWidget(browseButton,0);
    fl->addRow(hl);

    int lastCompletedExperiment = ui->exptValueLabel->text().toInt();
    if(p_batchManager && !p_batchManager->isComplete()
            && d_currentExptNum == lastCompletedExperiment)
        lastCompletedExperiment--;

    if(lastCompletedExperiment < 1)
    {
        numBox->setRange(0,__INT_MAX__);
        numBox->setSpecialValueText(QString("Select..."));
        numBox->setEnabled(false);
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
           numBox->setEnabled(false);
       }
       else
       {
           numBox->setEnabled(true);
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

        // Get the full path for tracking
        QString fullPath = BlackchirpCSV::exptDir(num, path).absolutePath();
        
        // Check if experiment is already open
        auto it = d_openExperiments.find(fullPath);
        if (it != d_openExperiments.end()) {
            // Experiment already open, raise existing window
            ExperimentViewWidget* existingWidget = it->second.get();
            existingWidget->show();
            existingWidget->raise();
            existingWidget->notifyAlreadyOpen();
            return;
        }
        
        // Create new experiment view widget
        auto evw = std::make_unique<ExperimentViewWidget>(num, path, true);
        ExperimentViewWidget* evwPtr = evw.get();
        
        // Connect signals for cleanup and window management
        connect(this, &MainWindow::closing, evwPtr, &ExperimentViewWidget::close);
        connect(evwPtr, &ExperimentViewWidget::widgetClosing, this, [this, fullPath]() {
            removeExperimentWidget(fullPath);
        });
        
        // Store in tracking map and show
        d_openExperiments[fullPath] = std::move(evw);
        updateViewExperimentMenu();
        evwPtr->show();
        evwPtr->raise();
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

QWidget *MainWindow::wrapWithPythonWidget(const QString &hwKey, QWidget *typeWidget)
{
    auto composite = new QWidget;
    auto layout = new QVBoxLayout(composite);
    layout->setContentsMargins(0, 0, 0, 0);

    auto pyWidget = new PythonHardwareControlWidget(hwKey, p_hwm, composite);
    layout->addWidget(pyWidget);

    if (typeWidget) {
        typeWidget->setParent(composite);
        layout->addWidget(typeWidget);
    }

    return composite;
}

HWDialog *MainWindow::createHWDialog(const QString key, QWidget *controlWidget)
{
    auto out = new HWDialog(key,controlWidget);
    d_openDialogs.insert({key,out});
    connect(out,&HWDialog::accepted,[this,key](){
        QMetaObject::invokeMethod(p_hwm,[this,key](){ p_hwm->updateObjectSettings(key); });
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
    
    // REPLACE binary logic with fine-grained critical hardware checking:
    if(!isCriticalHardwareConnected())
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
    ui->appConfigAction->setEnabled(false);

    switch(s)
    {
    case Asleep:
        ui->sleepButton->setEnabled(true);
        ui->appConfigAction->setEnabled(true);
        break;
    case Disconnected:
        ui->actionCommunication->setEnabled(true);
        ui->actionTest_All_Connections->setEnabled(true);
        ui->actionRuntimeHardwareConfig->setEnabled(true);
        ui->appConfigAction->setEnabled(true);
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
                if(act == ui->actionFtmwConfig)
                    continue;
                if(act == ui->actionLifConfig)
                    continue;
                if(act == ui->actionCommunication)
                    continue;
                if(act == ui->actionTest_All_Connections)
                    continue;
                if(act == ui->actionRuntimeHardwareConfig)
                    continue;
                if(act == ui->menuLoadout->menuAction())
                    continue;
                if(act == ui->menuFtmwPreset->menuAction())
                    continue;
                act->setEnabled(true);
            }
        }
        break;
    case Idle:
    default:
        for(auto act : hwl)
            act->setEnabled(true);
        if(ui->menuFtmwPreset->isEmpty())
            ui->menuFtmwPreset->menuAction()->setEnabled(false);
        for(auto act : acq)
            act->setEnabled(true);
        ui->sleepButton->setEnabled(true);
        ui->appConfigAction->setEnabled(true);
        break;
    }
}

void MainWindow::startBatch(BatchManager *bm)
{
    // Prompt user about overlays before committing to experiment start
    if (!ui->ftViewWidget->promptOverlayTransition()) {
        delete bm;
        return;
    }

    delete p_batchManager;

    connect(bm,&BatchManager::statusMessage,ui->statusBar,&QStatusBar::showMessage);
    connect(bm,&BatchManager::logMessage,p_lh,&LogHandler::logMessage);
    connect(bm,&BatchManager::beginExperiment,p_lh,&LogHandler::endExperimentLog);
    connect(bm,&BatchManager::beginExperiment,[this,bm](){
        QMetaObject::invokeMethod(p_hwm,[this,bm](){ p_hwm->initializeExperiment(bm->currentExperiment());});
    });
    connect(p_am,&AcquisitionManager::experimentComplete,bm,&BatchManager::experimentComplete);
    connect(p_am,&AcquisitionManager::experimentComplete,ui->ftViewWidget,&FtmwViewWidget::experimentComplete);
    connect(ui->abortButton,&QToolButton::clicked,bm,&BatchManager::abort);
    connect(bm,&BatchManager::batchComplete,this,&MainWindow::batchComplete);
    connect(bm,&BatchManager::batchComplete,this,&MainWindow::checkSleep);
    connect(bm,&BatchManager::batchComplete,p_lh,&LogHandler::endExperimentLog);

    if(ui->lifDisplayWidget)
        connect(p_am,&AcquisitionManager::experimentComplete,ui->lifDisplayWidget,&LifDisplayWidget::experimentComplete);

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

void MainWindow::removeExperimentWidget(const QString& path)
{
    auto it = d_openExperiments.find(path);
    if (it != d_openExperiments.end()) {
        d_openExperiments.erase(it);
        updateViewExperimentMenu();
    }
}

void MainWindow::updateViewExperimentMenu()
{
    // Get current actions (skip first action and separator)
    QList<QAction*> actions = ui->viewExperimentMenu->actions();
    
    // Remove actions for experiments that are no longer open
    // Start from index 2 to skip "View Experiment..." action and separator
    for (int i = actions.size() - 1; i >= 2; --i) {
        QAction* action = actions[i];
        QString actionPath = action->data().toString();
        
        if (d_openExperiments.find(actionPath) == d_openExperiments.end()) {
            ui->viewExperimentMenu->removeAction(action);
            action->deleteLater();
        }
    }
    
    // Add actions for new experiments that aren't in the menu yet
    for (const auto& [path, widget] : d_openExperiments) {
        bool actionExists = false;
        for (int i = 2; i < actions.size(); ++i) {
            if (actions[i]->data().toString() == path) {
                actionExists = true;
                break;
            }
        }
        
        if (!actionExists) {
            QString experimentTitle = widget->windowTitle();
            QAction* experimentAction = new QAction(experimentTitle, this);
            experimentAction->setData(path); // Store path in data for identification
            connect(experimentAction, &QAction::triggered, this, [this, path]() {
                showExistingExperiment(path);
            });
            ui->viewExperimentMenu->addAction(experimentAction);
        }
    }
}

void MainWindow::showExistingExperiment(const QString& path)
{
    auto it = d_openExperiments.find(path);
    if (it != d_openExperiments.end()) {
        ExperimentViewWidget* widget = it->second.get();
        widget->show();
        widget->raise();
        widget->notifyAlreadyOpen();
    }
}

void MainWindow::setupThemeAwareIconStyling()
{
    // Set BlackChirp branding - main application window icon  
    this->setWindowIcon(ThemeColors::createThemedIcon(":/icons/bc_logo_trans.svg", ThemeColors::IconPrimary, this));
    
    // Create theme-aware icons for control buttons using SVG color replacement
    ui->pauseButton->setIcon(ThemeColors::createThemedIcon(":/icons/pause.svg", ThemeColors::IconPrimary, this));
    ui->resumeButton->setIcon(ThemeColors::createThemedIcon(":/icons/play.svg", ThemeColors::IconPrimary, this));
    ui->abortButton->setIcon(ThemeColors::createThemedIcon(":/icons/stop.svg", ThemeColors::IconPrimary, this));
    ui->sleepButton->setIcon(ThemeColors::createThemedIcon(":/icons/moon.svg", ThemeColors::IconPrimary, this));
    
    // Set action icons
    ui->actionStart_Experiment->setIcon(ThemeColors::createThemedIcon(":/icons/document-plus.svg", ThemeColors::IconPrimary, this));
    ui->actionCommunication->setIcon(ThemeColors::createThemedIcon(":/icons/computer-desktop.svg", ThemeColors::IconPrimary, this));
    ui->actionTest_All_Connections->setIcon(ThemeColors::createThemedIcon(":/icons/link.svg", ThemeColors::IconPrimary, this));
    ui->actionQuick_Experiment->setIcon(ThemeColors::createThemedIcon(":/icons/quickexpt.svg", ThemeColors::IconPrimary, this));
    ui->actionStart_Sequence->setIcon(ThemeColors::createThemedIcon(":/icons/sequence.svg", ThemeColors::IconPrimary, this));
    
    if(ui->actionLifConfig)
        ui->actionLifConfig->setIcon(ThemeColors::createThemedIcon(":/icons/lif.svg", ThemeColors::IconPrimary, this));
    ui->actionRuntimeHardwareConfig->setIcon(ThemeColors::createThemedIcon(":/icons/cpu-chip.svg", ThemeColors::IconPrimary, this));

    ui->actionFtmwConfig->setIcon(ThemeColors::createThemedIcon(":/icons/rf.svg", ThemeColors::IconPrimary, this));
    
    // Set button icons  
    ui->acquireButton->setIcon(ThemeColors::createThemedIcon(":/icons/play-circle.svg", ThemeColors::IconPrimary, this));
    ui->hardwareButton->setIcon(ThemeColors::createThemedIcon(":/icons/wrench-screwdriver.svg", ThemeColors::IconPrimary, this));
    ui->settingsButton->setIcon(ThemeColors::createThemedIcon(":/icons/cog-6-tooth.svg", ThemeColors::IconSecondary, this));
    ui->helpButton->setIcon(ThemeColors::createThemedIcon(":/icons/question-mark-circle.svg", ThemeColors::IconSecondary, this));
    ui->auxPlotButton->setIcon(ThemeColors::createThemedIcon(":/icons/chart-bar.svg", ThemeColors::IconSecondary, this));
    ui->rollingPlotButton->setIcon(ThemeColors::createThemedIcon(":/icons/arrow-path-rounded-square.svg", ThemeColors::IconSecondary, this));
    ui->viewExperimentButton->setIcon(ThemeColors::createThemedIcon(":/icons/viewold.svg", ThemeColors::IconSecondary, this));
    
    // Set action icons
    ui->viewExperimentAction->setIcon(ThemeColors::createThemedIcon(":/icons/viewold.svg", ThemeColors::IconSecondary, this));
    ui->appConfigAction->setIcon(ThemeColors::createThemedIcon(":/icons/cog-6-tooth.svg", ThemeColors::IconSecondary, this));
    ui->exptConfigButton->setIcon(ThemeColors::createThemedIcon(":/icons/cog-6-tooth.svg", ThemeColors::IconSecondary, this));
    
    // Set tab icons
    ui->mainTabWidget->setTabIcon(ui->mainTabWidget->indexOf(ui->ftmwTab), ThemeColors::createThemedIcon(":/icons/signal.svg", ThemeColors::IconPrimary, this));
    ui->mainTabWidget->setTabIcon(ui->mainTabWidget->indexOf(ui->rollingDataTab), ThemeColors::createThemedIcon(":/icons/arrow-path-rounded-square.svg", ThemeColors::IconSecondary, this));
    ui->mainTabWidget->setTabIcon(ui->mainTabWidget->indexOf(ui->auxDataTab), ThemeColors::createThemedIcon(":/icons/chart-bar.svg", ThemeColors::IconSecondary, this));
    if(ui->lifTab)
        ui->mainTabWidget->setTabIcon(ui->mainTabWidget->indexOf(ui->lifTab), ThemeColors::createThemedIcon(":/icons/sparkles.svg", ThemeColors::IconSecondary, this));
    
    // Set autoscale action icons
    ui->actionAutoscale_Rolling->setIcon(ThemeColors::createThemedIcon(":/icons/arrows-pointing-out.svg", ThemeColors::IconSecondary, this));
    ui->actionAutoscale_Aux->setIcon(ThemeColors::createThemedIcon(":/icons/arrows-pointing-out.svg", ThemeColors::IconSecondary, this));
    
    // Set menu icons
    ui->menuRollingData->setIcon(ThemeColors::createThemedIcon(":/icons/arrow-path-rounded-square.svg", ThemeColors::IconSecondary, this));
    ui->menuAuxData->setIcon(ThemeColors::createThemedIcon(":/icons/chart-bar.svg", ThemeColors::IconSecondary, this));
}

std::shared_ptr<Experiment> MainWindow::createExperiment()
{
    auto exp = std::make_shared<Experiment>();
    // Populate hardware data from RuntimeHardwareConfig
    exp->d_hardwareData = RuntimeHardwareConfig::constInstance().createHardwareDataContainer();
    return exp;
}

void MainWindow::updateSavePathLabel()
{
    SettingsStorage s;
    d_savePath = s.get(BC::Key::savePath, QString(""));
    ui->savePathLabel->setToolTip(d_savePath);
    const QFontMetrics fm(ui->savePathLabel->font());
    const int w = ui->savePathLabel->width();
    ui->savePathLabel->setText(fm.elidedText(d_savePath, Qt::ElideMiddle, w > 0 ? w : 200));
}

bool MainWindow::eventFilter(QObject *watched, QEvent *event)
{
    if(watched == ui->savePathLabel && event->type() == QEvent::Resize)
    {
        const QFontMetrics fm(ui->savePathLabel->font());
        ui->savePathLabel->setText(fm.elidedText(d_savePath, Qt::ElideMiddle, ui->savePathLabel->width()));
    }
    else if(watched == ui->savePathLabel && event->type() == QEvent::Enter)
    {
        setCursor(Qt::PointingHandCursor);
        auto f = ui->savePathLabel->font();
        f.setUnderline(true);
        ui->savePathLabel->setFont(f);
    }
    else if(watched == ui->savePathLabel && event->type() == QEvent::Leave)
    {
        unsetCursor();
        auto f = ui->savePathLabel->font();
        f.setUnderline(false);
        ui->savePathLabel->setFont(f);
    }
    else if(watched == ui->savePathLabel && event->type() == QEvent::MouseButtonRelease)
    {
        auto me = static_cast<QMouseEvent*>(event);
        
        if(me->button() == Qt::LeftButton && 
        ui->savePathLabel->rect().contains(me->pos()))
            QDesktopServices::openUrl(QUrl::fromLocalFile(d_savePath));
    }
    return QMainWindow::eventFilter(watched, event);
}
