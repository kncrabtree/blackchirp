#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "communicationdialog.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow), d_hardwareConnected(false), d_state(Idle)
{
    ui->setupUi(this);

    QLabel *statusLabel = new QLabel(this);
    connect(this,&MainWindow::statusMessage,statusLabel,&QLabel::setText);
    ui->statusBar->addWidget(statusLabel);


    p_lh = new LogHandler();
    connect(p_lh,&LogHandler::sendLogMessage,ui->log,&QTextEdit::append);

    QThread *lhThread = new QThread(this);
    connect(lhThread,&QThread::finished,p_lh,&LogHandler::deleteLater);
    p_lh->moveToThread(lhThread);
    d_threadObjectList.append(qMakePair(lhThread,p_lh));
    lhThread->start();

    p_hwm = new HardwareManager();
    connect(p_hwm,&HardwareManager::logMessage,p_lh,&LogHandler::logMessage);
    connect(p_hwm,&HardwareManager::statusMessage,statusLabel,&QLabel::setText);
    connect(p_hwm,&HardwareManager::experimentInitialized,this,&MainWindow::experimentInitialized);
    connect(p_hwm,&HardwareManager::allHardwareConnected,this,&MainWindow::hardwareInitialized);

    QThread *hwmThread = new QThread(this);
    connect(hwmThread,&QThread::started,p_hwm,&HardwareManager::initialize);
    connect(hwmThread,&QThread::finished,p_hwm,&HardwareManager::deleteLater);
    p_hwm->moveToThread(hwmThread);
    d_threadObjectList.append(qMakePair(hwmThread,p_hwm));


    p_am = new AcquisitionManager();
    connect(p_am,&AcquisitionManager::logMessage,p_lh,&LogHandler::logMessage);
    connect(p_am,&AcquisitionManager::statusMessage,statusLabel,&QLabel::setText);
    connect(p_am,&AcquisitionManager::ftmwShotAcquired,ui->ftmwProgressBar,&QProgressBar::setValue);
    connect(ui->actionPause,&QAction::triggered,p_am,&AcquisitionManager::pause);
    connect(ui->actionResume,&QAction::triggered,p_am,&AcquisitionManager::resume);
    connect(ui->actionAbort,&QAction::triggered,p_am,&AcquisitionManager::abort);

    connect(p_am,&AcquisitionManager::newFidList,ui->ftViewWidget,&FtmwViewWidget::newFidList);

    QThread *amThread = new QThread(this);
    connect(amThread,&QThread::finished,p_am,&AcquisitionManager::deleteLater);
    p_am->moveToThread(amThread);
    d_threadObjectList.append(qMakePair(amThread,p_am));

    connect(p_hwm,&HardwareManager::experimentInitialized,p_am,&AcquisitionManager::beginExperiment);
    connect(p_hwm,&HardwareManager::scopeShotAcquired,p_am,&AcquisitionManager::processScopeShot);
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

    //build experiment from a wizard or something
    Experiment e;
    FtmwConfig ft;
    ft.setEnabled();
    ft.setTargetShots(100);
    ft.setType(FtmwConfig::TargetShots);
    FtmwConfig::ScopeConfig sc = ft.scopeConfig();
    sc.bytesPerPoint = 2;
    sc.vScale = 0.02;
    sc.recordLength = 750000;
    sc.sampleRate = 50e9;
    sc.numFrames = 10;
    sc.fastFrameEnabled = true;
    sc.summaryFrame = false;
    ft.setScopeConfig(sc);
    e.setFtmwConfig(ft);
    e.setTimeDataInterval(1);

    BatchSingle *bs = new BatchSingle(e);

    startBatch(bs);
}

void MainWindow::batchComplete(bool aborted)
{
    if(aborted)
        emit statusMessage(QString("Experiment aborted"));
    else
        emit statusMessage(QString("Experiment complete"));

    if(ui->ftmwProgressBar->maximum() == 0)
    {
	    ui->ftmwProgressBar->setRange(0,1);
	    ui->ftmwProgressBar->setValue(1);
    }

    configureUi(Idle);
}

void MainWindow::experimentInitialized(Experiment exp)
{
	if(!exp.isInitialized())
		return;

	ui->exptSpinBox->setValue(exp.number());

	if(exp.ftmwConfig().isEnabled())
	{
		if(exp.ftmwConfig().type() != FtmwConfig::TargetShots)
			ui->ftmwProgressBar->setRange(0,0);
		else
		{
			ui->ftmwProgressBar->setRange(0,exp.ftmwConfig().targetShots());
			ui->ftmwProgressBar->setValue(0);
		}
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
        break;
    case Disconnected:
        ui->actionAbort->setEnabled(false);
        ui->actionPause->setEnabled(false);
        ui->actionResume->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(false);
        ui->actionCommunication->setEnabled(true);
        break;
    case Paused:
        ui->actionAbort->setEnabled(true);
        ui->actionPause->setEnabled(false);
        ui->actionResume->setEnabled(true);
        ui->actionStart_Experiment->setEnabled(false);
        ui->actionCommunication->setEnabled(false);
        break;
    case Acquiring:
        ui->actionAbort->setEnabled(true);
        ui->actionPause->setEnabled(true);
        ui->actionResume->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(false);
        ui->actionCommunication->setEnabled(false);
        break;
    case Idle:
    default:
        ui->actionAbort->setEnabled(false);
        ui->actionPause->setEnabled(false);
        ui->actionResume->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(true);
        ui->actionCommunication->setEnabled(true);
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

    if(sleepWhenDone)
    {
        //connect to sleep action
    }

    configureUi(Acquiring);
    bm->moveToThread(d_batchThread);
    d_batchThread->start();
}
