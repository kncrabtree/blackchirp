#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "communicationdialog.h"
#include "ftmwconfigwidget.h"
#include <QDialogButtonBox>
#include <QPushButton>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow), d_hardwareConnected(false), d_state(Idle)
{
    ui->setupUi(this);

    ui->exptSpinBox->blockSignals(true);
    ui->valonTXDoubleSpinBox->blockSignals(true);
    ui->valonRXDoubleSpinBox->blockSignals(true);

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
    connect(p_hwm,&HardwareManager::valonTxFreqRead,ui->valonTXDoubleSpinBox,&QDoubleSpinBox::setValue);
    connect(p_hwm,&HardwareManager::valonRxFreqRead,ui->valonRXDoubleSpinBox,&QDoubleSpinBox::setValue);

    QThread *hwmThread = new QThread(this);
    connect(hwmThread,&QThread::started,p_hwm,&HardwareManager::initialize);
    connect(hwmThread,&QThread::finished,p_hwm,&HardwareManager::deleteLater);
    p_hwm->moveToThread(hwmThread);
    d_threadObjectList.append(qMakePair(hwmThread,p_hwm));


    p_am = new AcquisitionManager();
    connect(p_am,&AcquisitionManager::logMessage,p_lh,&LogHandler::logMessage);
    connect(p_am,&AcquisitionManager::statusMessage,statusLabel,&QLabel::setText);
    connect(p_am,&AcquisitionManager::ftmwShotAcquired,ui->ftmwProgressBar,&QProgressBar::setValue);
    connect(p_am,&AcquisitionManager::ftmwShotAcquired,ui->ftViewWidget,&FtmwViewWidget::updateShotsLabel);
    connect(ui->actionPause,&QAction::triggered,p_am,&AcquisitionManager::pause);
    connect(ui->actionResume,&QAction::triggered,p_am,&AcquisitionManager::resume);
    connect(ui->actionAbort,&QAction::triggered,p_am,&AcquisitionManager::abort);
    connect(ui->ftViewWidget,&FtmwViewWidget::rollingAverageShotsChanged,p_am,&AcquisitionManager::changeRollingAverageShots);
    connect(ui->ftViewWidget,&FtmwViewWidget::rollingAverageReset,p_am,&AcquisitionManager::resetRollingAverage);

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
    connect(ui->actionTrackingShow,&QAction::triggered,[=](){ ui->tabWidget->setCurrentIndex(2); });
    connect(ui->action_Graphs,&QAction::triggered,ui->trackingViewWidget,&TrackingViewWidget::changeNumPlots);

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

    QDialog d;
    d.setWindowTitle(QString("Configure FTMW Acquisition"));
    QVBoxLayout *vbl = new QVBoxLayout;
    FtmwConfigWidget *w = new FtmwConfigWidget;
    vbl->addWidget(w);

    QDialogButtonBox *bb = new QDialogButtonBox(QDialogButtonBox::Ok|QDialogButtonBox::Cancel|QDialogButtonBox::Reset);
    connect(bb->button(QDialogButtonBox::Ok),&QPushButton::clicked,w,&FtmwConfigWidget::saveToSettings);
    connect(bb->button(QDialogButtonBox::Ok),&QPushButton::clicked,&d,&QDialog::accept);
    connect(bb->button(QDialogButtonBox::Cancel),&QPushButton::clicked,&d,&QDialog::reject);
    connect(bb->button(QDialogButtonBox::Reset),&QPushButton::clicked,w,&FtmwConfigWidget::loadFromSettings);
    vbl->addWidget(bb);

    d.setLayout(vbl);
    int ret = d.exec();

    if(ret == QDialog::Rejected)
        return;


    e.setFtmwConfig(w->getConfig());
    e.setTimeDataInterval(5);

    BatchSingle *bs = new BatchSingle(e);

    startBatch(bs);
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

    configureUi(Idle);
}

void MainWindow::experimentInitialized(Experiment exp)
{
	if(!exp.isInitialized())
		return;

	ui->exptSpinBox->setValue(exp.number());
    ui->ftmwProgressBar->setValue(0);
    ui->ftViewWidget->initializeForExperiment(exp.ftmwConfig());

	if(exp.ftmwConfig().isEnabled())
	{
        switch(exp.ftmwConfig().type()) {
        case FtmwConfig::TargetShots:
            ui->ftmwProgressBar->setRange(0,exp.ftmwConfig().targetShots());
            break;
        case FtmwConfig::TargetTime:
            ui->ftmwProgressBar->setRange(0,static_cast<int>(exp.startTime().secsTo(exp.ftmwConfig().targetTime())));
            break;
        default:
			ui->ftmwProgressBar->setRange(0,0);
            break;
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

void MainWindow::closeEvent(QCloseEvent *ev)
{
    if(d_batchThread->isRunning())
        ev->ignore();
    else
        ev->accept();
}
