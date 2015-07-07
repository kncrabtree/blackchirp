#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QThread>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QCloseEvent>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QLineEdit>

#include "communicationdialog.h"
#include "ftmwconfigwidget.h"
#include "rfconfigwidget.h"
#include "experimentwizard.h"
#include "loghandler.h"
#include "hardwaremanager.h"
#include "acquisitionmanager.h"
#include "batchmanager.h"
#include "led.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow), d_hardwareConnected(false), d_state(Idle), d_logCount(0), d_logIcon(BlackChirp::LogNormal)
{
    ui->setupUi(this);

    ui->exptSpinBox->blockSignals(true);
    ui->valonTXDoubleSpinBox->blockSignals(true);
    ui->valonRXDoubleSpinBox->blockSignals(true);

    auto vc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);

    QLabel *statusLabel = new QLabel(this);
    connect(this,&MainWindow::statusMessage,statusLabel,&QLabel::setText);
    ui->statusBar->addWidget(statusLabel);

    p_lh = new LogHandler();
    connect(p_lh,&LogHandler::sendLogMessage,ui->log,&QTextEdit::append);
    connect(p_lh,&LogHandler::iconUpdate,this,&MainWindow::setLogIcon);
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
    connect(p_lh,&LogHandler::sendLogMessage,[=](){
        if(ui->tabWidget->currentWidget() != ui->logTab)
        {
            d_logCount++;
            ui->tabWidget->setTabText(ui->tabWidget->indexOf(ui->logTab),QString("Log (%1)").arg(d_logCount));
        }
    });

    QGridLayout *gl = new QGridLayout;
    for(int i=0; i<BC_PGEN_NUMCHANNELS; i++)
    {
        QLabel *lbl = new QLabel(QString("Ch%1").arg(i),this);
        lbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);

        Led *led = new Led(this);
        gl->addWidget(lbl,i/2,(2*i)%4,1,1,Qt::AlignVCenter);
        gl->addWidget(led,i/2,((2*i)%4)+1,1,1,Qt::AlignVCenter);

        d_ledList.append(qMakePair(lbl,led));
    }
    gl->setColumnStretch(0,1);
    gl->setColumnStretch(1,0);
    gl->setColumnStretch(2,1);
    gl->setColumnStretch(3,0);
    ui->pulseConfigBox->setLayout(gl);


    QThread *lhThread = new QThread(this);
    connect(lhThread,&QThread::finished,p_lh,&LogHandler::deleteLater);
    p_lh->moveToThread(lhThread);
    d_threadObjectList.append(qMakePair(lhThread,p_lh));
    lhThread->start();

    p_hwm = new HardwareManager();
    connect(p_hwm,&HardwareManager::logMessage,p_lh,&LogHandler::logMessage);
    connect(p_hwm,&HardwareManager::statusMessage,statusLabel,&QLabel::setText);
    connect(p_hwm,&HardwareManager::hwInitializationComplete,ui->pulseConfigWidget,&PulseConfigWidget::updateHardwareLimits);
    connect(p_hwm,&HardwareManager::hwInitializationComplete,ui->lifControlWidget,&LifControlWidget::updateHardwareLimits);
    connect(p_hwm,&HardwareManager::experimentInitialized,this,&MainWindow::experimentInitialized);
    connect(p_hwm,&HardwareManager::allHardwareConnected,this,&MainWindow::hardwareInitialized);
    connect(p_hwm,&HardwareManager::valonTxFreqRead,ui->valonTXDoubleSpinBox,&QDoubleSpinBox::setValue);
    connect(p_hwm,&HardwareManager::valonRxFreqRead,ui->valonRXDoubleSpinBox,&QDoubleSpinBox::setValue);
    connect(p_hwm,&HardwareManager::pGenConfigUpdate,ui->pulseConfigWidget,&PulseConfigWidget::newConfig);
    connect(p_hwm,&HardwareManager::pGenSettingUpdate,ui->pulseConfigWidget,&PulseConfigWidget::newSetting);
    connect(p_hwm,&HardwareManager::pGenConfigUpdate,this,&MainWindow::updatePulseLeds);
    connect(p_hwm,&HardwareManager::pGenSettingUpdate,this,&MainWindow::updatePulseLed);
    connect(p_hwm,&HardwareManager::flowUpdate,this,&MainWindow::updateFlow);
    connect(p_hwm,&HardwareManager::flowNameUpdate,this,&MainWindow::updateFlowName);
    connect(p_hwm,&HardwareManager::flowSetpointUpdate,this,&MainWindow::updateFlowSetpoint);
    connect(p_hwm,&HardwareManager::pressureUpdate,ui->pressureDoubleSpinBox,&QDoubleSpinBox::setValue);
    connect(p_hwm,&HardwareManager::pressureSetpointUpdate,this,&MainWindow::updatePressureSetpoint);
    connect(p_hwm,&HardwareManager::pressureControlMode,this,&MainWindow::updatePressureControl);
    connect(ui->pressureControlButton,&QPushButton::clicked,p_hwm,&HardwareManager::setPressureControlMode);
    connect(ui->pressureControlBox,vc,p_hwm,&HardwareManager::setPressureSetpoint);
    connect(p_hwm,&HardwareManager::pGenRepRateUpdate,ui->pulseConfigWidget,&PulseConfigWidget::newRepRate);
    connect(ui->pulseConfigWidget,&PulseConfigWidget::changeSetting,p_hwm,&HardwareManager::setPGenSetting);
    connect(ui->pulseConfigWidget,&PulseConfigWidget::changeRepRate,p_hwm,&HardwareManager::setPGenRepRate);
    connect(p_hwm,&HardwareManager::lifScopeShotAcquired,ui->lifControlWidget,&LifControlWidget::newTrace);
    connect(p_hwm,&HardwareManager::lifScopeConfigUpdated,ui->lifControlWidget,&LifControlWidget::scopeConfigChanged);
    connect(ui->lifControlWidget,&LifControlWidget::updateScope,p_hwm,&HardwareManager::setLifScopeConfig);
    connect(p_hwm,&HardwareManager::lifSettingsComplete,ui->lifDisplayWidget,&LifDisplayWidget::resetLifPlot);

    QThread *hwmThread = new QThread(this);
    connect(hwmThread,&QThread::started,p_hwm,&HardwareManager::initialize);
    connect(hwmThread,&QThread::finished,p_hwm,&HardwareManager::deleteLater);
    p_hwm->moveToThread(hwmThread);
    d_threadObjectList.append(qMakePair(hwmThread,p_hwm));

    gl = static_cast<QGridLayout*>(ui->gasControlBox->layout());
    QGridLayout *gl2 = static_cast<QGridLayout*>(ui->flowStatusBox->layout());
    for(int i=0; i<BC_FLOW_NUMCHANNELS; i++)
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

        fw.nameLabel = new QLabel(QString("Ch%1").arg(i+1),this);
        fw.nameLabel->setMinimumWidth(QFontMetrics(QFont(QString("sans-serif"))).width(QString("MMMMMMMM")));
        fw.nameLabel->setAlignment(Qt::AlignRight|Qt::AlignVCenter);

        fw.led = new Led(this);

        fw.displayBox = new QDoubleSpinBox(this);
        fw.displayBox->setRange(-9999.9,9999.9);
        fw.displayBox->setDecimals(1);
        fw.displayBox->setSuffix(QString(" sccm"));
        fw.displayBox->blockSignals(true);
        fw.displayBox->setReadOnly(true);
        fw.displayBox->setButtonSymbols(QAbstractSpinBox::NoButtons);

        d_flowWidgets.append(fw);

        gl->addWidget(new QLabel(QString::number(i+1),this),1+i,0,1,1);
        gl->addWidget(fw.nameEdit,i+1,1,1,1);
        gl->addWidget(fw.controlBox,i+1,2,1,1);

        gl2->addWidget(fw.nameLabel,i+1,0,1,1,Qt::AlignRight);
        gl2->addWidget(fw.displayBox,i+1,1,1,1);
        gl2->addWidget(fw.led,i+1,2,1,1);
    }
    gl->addWidget(new QLabel(QString("Pressure"),this),2+BC_FLOW_NUMCHANNELS,1,1,1,Qt::AlignRight);
    gl->addWidget(ui->pressureControlBox,2+BC_FLOW_NUMCHANNELS,2,1,1);
    gl->addWidget(new QLabel(QString("Pressure Control Mode"),this),3+BC_FLOW_NUMCHANNELS,1,1,1,Qt::AlignRight);
    gl->addWidget(ui->pressureControlButton,3+BC_FLOW_NUMCHANNELS,2,1,1);
    gl->addItem(new QSpacerItem(10,10,QSizePolicy::Minimum,QSizePolicy::Expanding),4+BC_FLOW_NUMCHANNELS,0,1,3);

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
    connect(p_am,&AcquisitionManager::ftmwShotAcquired,ui->ftmwProgressBar,&QProgressBar::setValue);
    connect(p_am,&AcquisitionManager::ftmwShotAcquired,ui->ftViewWidget,&FtmwViewWidget::updateShotsLabel);
    connect(p_am,&AcquisitionManager::lifShotAcquired,ui->lifProgressBar,&QProgressBar::setValue);
    connect(ui->actionPause,&QAction::triggered,p_am,&AcquisitionManager::pause);
    connect(ui->actionResume,&QAction::triggered,p_am,&AcquisitionManager::resume);
    connect(ui->actionAbort,&QAction::triggered,p_am,&AcquisitionManager::abort);
    connect(ui->ftViewWidget,&FtmwViewWidget::rollingAverageShotsChanged,p_am,&AcquisitionManager::changeRollingAverageShots);
    connect(ui->ftViewWidget,&FtmwViewWidget::rollingAverageReset,p_am,&AcquisitionManager::resetRollingAverage);
    connect(p_am,&AcquisitionManager::newFidList,ui->ftViewWidget,&FtmwViewWidget::newFidList);
    connect(p_am,&AcquisitionManager::lifPointUpdate,ui->lifDisplayWidget,&LifDisplayWidget::updatePoint);

    QThread *amThread = new QThread(this);
    connect(amThread,&QThread::finished,p_am,&AcquisitionManager::deleteLater);
    p_am->moveToThread(amThread);
    d_threadObjectList.append(qMakePair(amThread,p_am));

    connect(p_hwm,&HardwareManager::experimentInitialized,p_am,&AcquisitionManager::beginExperiment);
    connect(p_hwm,&HardwareManager::ftmwScopeShotAcquired,p_am,&AcquisitionManager::processFtmwScopeShot);
    connect(p_am,&AcquisitionManager::nextLifPoint,p_hwm,&HardwareManager::setLifParameters);
    connect(p_hwm,&HardwareManager::lifSettingsComplete,p_am,&AcquisitionManager::lifHardwareReady);
    connect(p_hwm,&HardwareManager::lifScopeShotAcquired,p_am,&AcquisitionManager::processLifScopeShot);
    connect(p_am,&AcquisitionManager::experimentComplete,p_hwm,&HardwareManager::endAcquisition);
    connect(p_am,&AcquisitionManager::beginAcquisition,p_hwm,&HardwareManager::beginAcquisition);
    connect(p_am,&AcquisitionManager::timeDataSignal,p_hwm,&HardwareManager::getTimeData);
    connect(p_hwm,&HardwareManager::timeData,p_am,&AcquisitionManager::processTimeData);


    hwmThread->start();
    amThread->start();

    d_batchThread = new QThread(this);

    connect(ui->actionStart_Experiment,&QAction::triggered,this,&MainWindow::startExperiment);
    connect(ui->actionPause,&QAction::triggered,this,&MainWindow::pauseUi);
    connect(ui->actionResume,&QAction::triggered,this,&MainWindow::resumeUi);
    connect(ui->actionCommunication,&QAction::triggered,this,&MainWindow::launchCommunicationDialog);
    connect(ui->actionRf_Configuration,&QAction::triggered,this,&MainWindow::launchRfConfigDialog);
    connect(ui->actionTrackingShow,&QAction::triggered,[=](){ ui->tabWidget->setCurrentIndex(2); });
    connect(ui->action_Graphs,&QAction::triggered,ui->trackingViewWidget,&TrackingViewWidget::changeNumPlots);
    connect(ui->actionTest_All_Connections,&QAction::triggered,p_hwm,&HardwareManager::testAll);

    connect(ui->lifControlWidget,&LifControlWidget::lifColorChanged,
            ui->lifDisplayWidget,&LifDisplayWidget::checkLifColors);
    connect(ui->lifDisplayWidget,&LifDisplayWidget::lifColorChanged,
            ui->lifControlWidget,&LifControlWidget::checkLifColors);


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

void MainWindow::startExperiment()
{
    if(d_batchThread->isRunning())
        return;

    ExperimentWizard wiz(this);
    wiz.setPulseConfig(ui->pulseConfigWidget->getConfig());
    wiz.setFlowConfig(getFlowConfig());
    connect(p_hwm,&HardwareManager::lifScopeShotAcquired,&wiz,&ExperimentWizard::newTrace);
    connect(p_hwm,&HardwareManager::lifScopeConfigUpdated,&wiz,&ExperimentWizard::scopeConfigChanged);
    connect(&wiz,&ExperimentWizard::updateScope,p_hwm,&HardwareManager::setLifScopeConfig);
    connect(&wiz,&ExperimentWizard::lifColorChanged,ui->lifControlWidget,&LifControlWidget::checkLifColors);
    connect(&wiz,&ExperimentWizard::lifColorChanged,ui->lifDisplayWidget,&LifDisplayWidget::checkLifColors);

    if(wiz.exec() != QDialog::Accepted)
        return;


    wiz.saveToSettings();
    BatchManager *bm = wiz.getBatchManager();

    startBatch(bm);
}

void MainWindow::batchComplete(bool aborted)
{
    disconnect(p_hwm,&HardwareManager::timeData,ui->trackingViewWidget,&TrackingViewWidget::pointUpdated);
    disconnect(p_am,&AcquisitionManager::timeData,ui->trackingViewWidget,&TrackingViewWidget::pointUpdated);

    if(aborted)
        emit statusMessage(QString("Experiment aborted"));
    else
        emit statusMessage(QString("Experiment complete"));

    if(ui->ftmwProgressBar->maximum() == 0)
    {
	    ui->ftmwProgressBar->setRange(0,1);
	    ui->ftmwProgressBar->setValue(1);
    }

    disconnect(p_hwm,&HardwareManager::lifScopeShotAcquired,
               ui->lifDisplayWidget,&LifDisplayWidget::lifShotAcquired);

    configureUi(Idle);
}

void MainWindow::experimentInitialized(Experiment exp)
{
	if(!exp.isInitialized())
		return;

	ui->exptSpinBox->setValue(exp.number());
    ui->ftmwProgressBar->setValue(0);
    ui->ftViewWidget->initializeForExperiment(exp.ftmwConfig());

    ui->lifProgressBar->setValue(0);

	if(exp.ftmwConfig().isEnabled())
	{
        switch(exp.ftmwConfig().type()) {
        case BlackChirp::FtmwTargetShots:
            ui->ftmwProgressBar->setRange(0,exp.ftmwConfig().targetShots());
            break;
        case BlackChirp::FtmwTargetTime:
            ui->ftmwProgressBar->setRange(0,static_cast<int>(exp.startTime().secsTo(exp.ftmwConfig().targetTime())));
            break;
        default:
			ui->ftmwProgressBar->setRange(0,0);
            break;
        }
	}
    else
    {
        ui->ftmwProgressBar->setRange(0,1);
        ui->ftmwProgressBar->setValue(1);
    }

    ui->lifDisplayWidget->prepareForExperiment(exp.lifConfig());
    if(exp.lifConfig().isEnabled())
    {
        ui->lifProgressBar->setRange(0,exp.lifConfig().totalShots());
        connect(p_hwm,&HardwareManager::lifScopeShotAcquired,
                ui->lifDisplayWidget,&LifDisplayWidget::lifShotAcquired,Qt::UniqueConnection);
    }
    else
    {
        ui->lifProgressBar->setRange(0,1);
        ui->lifProgressBar->setValue(1);
    }
}

void MainWindow::hardwareInitialized(bool success)
{
	d_hardwareConnected = success;
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

void MainWindow::launchRfConfigDialog()
{
    QDialog d(this);
    d.setWindowTitle(QString("Rf Configuration"));
    QVBoxLayout *vbl = new QVBoxLayout;
    RfConfigWidget *rfw = new RfConfigWidget(ui->valonTXDoubleSpinBox->value(),ui->valonRXDoubleSpinBox->value());
    QDialogButtonBox *bb = new QDialogButtonBox(QDialogButtonBox::Reset|QDialogButtonBox::Ok|QDialogButtonBox::Cancel);

    vbl->addWidget(rfw);
    vbl->addWidget(bb);
    d.setLayout(vbl);

    connect(bb->button(QDialogButtonBox::Reset),&QAbstractButton::clicked,rfw,&RfConfigWidget::loadFromSettings);
    connect(bb->button(QDialogButtonBox::Ok),&QAbstractButton::clicked,rfw,&RfConfigWidget::saveSettings);
    connect(bb->button(QDialogButtonBox::Ok),&QAbstractButton::clicked,&d,&QDialog::accept);
    connect(bb->button(QDialogButtonBox::Cancel),&QAbstractButton::clicked,&d,&QDialog::reject);

    connect(p_hwm,&HardwareManager::valonTxFreqRead,rfw,&RfConfigWidget::txFreqUpdate);
    connect(p_hwm,&HardwareManager::valonRxFreqRead,rfw,&RfConfigWidget::rxFreqUpdate);
    connect(rfw,&RfConfigWidget::setValonTx,p_hwm,&HardwareManager::setValonTxFreq);
    connect(rfw,&RfConfigWidget::setValonRx,p_hwm,&HardwareManager::setValonRxFreq);

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
    case BlackChirp::PulseName:
        d_ledList.at(index).first->setText(val.toString());
        break;
    case BlackChirp::PulseEnabled:
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
    if(ui->tabWidget->currentWidget() == ui->logTab)
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

void MainWindow::configureUi(MainWindow::ProgramState s)
{
    if(!d_hardwareConnected)
        s = Disconnected;

    switch(s)
    {
    case Asleep:
        ui->actionAbort->setEnabled(false);
        ui->actionPause->setEnabled(false);
        ui->actionResume->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(false);
        ui->actionCommunication->setEnabled(false);
        ui->actionTest_All_Connections->setEnabled(false);
        ui->gasControlBox->setEnabled(false);
        ui->pulseConfigWidget->setEnabled(false);
	   ui->lifControlWidget->setEnabled(false);
        break;
    case Disconnected:
        ui->actionAbort->setEnabled(false);
        ui->actionPause->setEnabled(false);
        ui->actionResume->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(false);
        ui->actionCommunication->setEnabled(true);
        ui->actionTest_All_Connections->setEnabled(true);
        ui->gasControlBox->setEnabled(false);
        ui->pulseConfigWidget->setEnabled(false);
	   ui->lifControlWidget->setEnabled(false);
        break;
    case Paused:
        ui->actionAbort->setEnabled(true);
        ui->actionPause->setEnabled(false);
        ui->actionResume->setEnabled(true);
        ui->actionStart_Experiment->setEnabled(false);
        ui->actionCommunication->setEnabled(false);
        ui->actionTest_All_Connections->setEnabled(false);
        ui->gasControlBox->setEnabled(false);
        ui->pulseConfigWidget->setEnabled(false);
	   ui->lifControlWidget->setEnabled(false);
        break;
    case Acquiring:
        ui->actionAbort->setEnabled(true);
        ui->actionPause->setEnabled(true);
        ui->actionResume->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(false);
        ui->actionCommunication->setEnabled(false);
        ui->actionTest_All_Connections->setEnabled(false);
        ui->gasControlBox->setEnabled(false);
        ui->pulseConfigWidget->setEnabled(false);
	   ui->lifControlWidget->setEnabled(false);
        break;
    case Idle:
    default:
        ui->actionAbort->setEnabled(false);
        ui->actionPause->setEnabled(false);
        ui->actionResume->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(true);
        ui->actionCommunication->setEnabled(true);
        ui->actionTest_All_Connections->setEnabled(true);
        ui->gasControlBox->setEnabled(true);
        ui->pulseConfigWidget->setEnabled(true);
	   ui->lifControlWidget->setEnabled(true);
        break;
    }

    if(s != Disconnected)
	    d_state = s;
}

void MainWindow::startBatch(BatchManager *bm, bool sleepWhenDone)
{
    connect(d_batchThread,&QThread::started,bm,&BatchManager::beginNextExperiment);
    connect(bm,&BatchManager::logMessage,p_lh,&LogHandler::logMessage);
    connect(bm,&BatchManager::beginExperiment,p_hwm,&HardwareManager::initializeExperiment);
    connect(p_am,&AcquisitionManager::experimentComplete,bm,&BatchManager::experimentComplete);
    connect(bm,&BatchManager::batchComplete,this,&MainWindow::batchComplete);
    connect(bm,&BatchManager::batchComplete,d_batchThread,&QThread::quit);
    connect(d_batchThread,&QThread::finished,bm,&BatchManager::deleteLater);

    connect(p_hwm,&HardwareManager::timeData,ui->trackingViewWidget,&TrackingViewWidget::pointUpdated,Qt::UniqueConnection);
    connect(p_am,&AcquisitionManager::timeData,ui->trackingViewWidget,&TrackingViewWidget::pointUpdated,Qt::UniqueConnection);

    if(sleepWhenDone)
    {
        //connect to sleep action
    }

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
        ev->accept();
}
