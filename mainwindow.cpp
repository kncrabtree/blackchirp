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

    QThread *lhThread = new QThread(this);
    p_lh->moveToThread(lhThread);
    d_threadList.append(qMakePair(lhThread,p_lh));
    lhThread->start();
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
