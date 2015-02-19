#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow), d_hardwareConnected(false)
{
    ui->setupUi(this);

    QLabel *statusLabel = new QLabel(this);
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

    QThread *hwmThread = new QThread(this);
    connect(hwmThread,&QThread::started,p_hwm,&HardwareManager::initialize);
    connect(hwmThread,&QThread::finished,p_hwm,&HardwareManager::deleteLater);
    p_hwm->moveToThread(hwmThread);
    d_threadObjectList.append(qMakePair(hwmThread,p_hwm));
    hwmThread->start();

    p_am = new AcquisitionManager();
    connect(p_am,&AcquisitionManager::logMessage,p_lh,&LogHandler::logMessage);
    connect(p_am,&AcquisitionManager::statusMessage,statusLabel,&QLabel::setText);

    QThread *amThread = new QThread(this);
    connect(amThread,&QThread::finished,p_am,&AcquisitionManager::deleteLater);
    p_am->moveToThread(amThread);
    d_threadObjectList.append(qMakePair(amThread,p_am));
    amThread->start();

    d_batchThread = new QThread(this);
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
        break;
    case Disconnected:
        ui->actionAbort->setEnabled(false);
        ui->actionPause->setEnabled(false);
        ui->actionResume->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(false);
        break;
    case Paused:
        ui->actionAbort->setEnabled(false);
        ui->actionPause->setEnabled(false);
        ui->actionResume->setEnabled(true);
        ui->actionStart_Experiment->setEnabled(false);
        break;
    case Acquiring:
        ui->actionAbort->setEnabled(true);
        ui->actionPause->setEnabled(true);
        ui->actionResume->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(false);
        break;
    case Idle:
    default:
        ui->actionAbort->setEnabled(false);
        ui->actionPause->setEnabled(false);
        ui->actionResume->setEnabled(false);
        ui->actionStart_Experiment->setEnabled(true);
        break;
    }
}

void MainWindow::startBatch(BatchManager *bm, bool sleepWhenDone)
{
    connect(d_batchThread,&QThread::started,bm,&BatchManager::beginNextExperiment);
    connect(bm,&BatchManager::logMessage,p_lh,&LogHandler::logMessage);
//    connect(bm,&BatchManager::beginExperiment,p_hwm,&HardwareManager::initializeExperiment);
//    connect(p_am,&AcquisitionManager::experimentComplete,bm,&BatchManager::experimentComplete);
//    connect(bm,&BatchManager::batchComplete,this,&MainWindow::batchComplete);
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
