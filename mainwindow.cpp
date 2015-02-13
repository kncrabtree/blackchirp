#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QLabel *statusLabel = new QLabel(this);
    ui->statusBar->addWidget(statusLabel);


    p_lh = new LogHandler();
    connect(p_lh,&LogHandler::sendStatusMessage,statusLabel,&QLabel::setText);
    connect(p_lh,&LogHandler::sendLogMessage,ui->log,&QTextEdit::append);

    QThread *lhThread = new QThread(this);
    p_lh->moveToThread(lhThread);
    d_threadList.append(qMakePair(lhThread,p_lh));
    lhThread->start();

    p_hwm = new HardwareManager();
    connect(p_hwm,&HardwareManager::logMessage,p_lh,&LogHandler::logMessage);
    connect(p_hwm,&HardwareManager::statusMessage,p_lh,&LogHandler::sendStatusMessage);

    QThread *hwmThread = new QThread(this);
    connect(hwmThread,&QThread::started,p_hwm,&HardwareManager::initialize);
    p_hwm->moveToThread(hwmThread);
    d_threadList.append(qMakePair(hwmThread,p_hwm));
    hwmThread->start();
}

MainWindow::~MainWindow()
{
    for(int i=0; i<d_threadList.size();i++)
    {
        QThread *t = d_threadList.at(i).first;
        QObject *obj = d_threadList.at(i).second;

        t->quit();
        t->wait();
        delete obj;
    }

    delete ui;
}
