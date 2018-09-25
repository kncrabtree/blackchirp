#include "ftmwviewwidget.h"

#include <QThread>
#include <QMessageBox>

#include "ftworker.h"
#include "ftmwsnapshotwidget.h"

FtmwViewWidget::FtmwViewWidget(QWidget *parent, QString path) :
    QWidget(parent),
    ui(new Ui::FtmwViewWidget), d_replotWhenDone(false), d_processing(false), d_currentExptNum(-1),
    d_mode(Live), d_frame1(0), d_frame2(0), d_segment1(0), d_segment2(0), d_path(path)
{
    ui->setupUi(this);

    d_currentProcessingSettings = FtWorker::FidProcessingSettings { -1.0, -1.0, 0, false, 1.0, 50.0, BlackChirp::Boxcar };

    p_liveThread = new QThread(this);
    d_threadList << p_liveThread;

    p_liveFtw = nullptr;

    p_mainFtw = new FtWorker(d_mainFtwId);
    auto *mthread = new QThread(this);
    d_threadList << mthread;
    p_mainFtw->moveToThread(mthread);

    p_plot1Ftw = new FtWorker(d_plot1FtwId);
    auto *p1thread = new QThread(this);
    d_threadList << p1thread;
    p_plot1Ftw->moveToThread(p1thread);

    p_plot2Ftw = new FtWorker(d_plot2FtwId);
    auto *p2thread = new QThread(this);
    d_threadList << p2thread;
    p_plot2Ftw->moveToThread(p2thread);

    ui->snapshotWidget1->hide();
    ui->snapshotWidget2->hide();

    ///TODO: load processing settings from settings
    ///TODO: Make signal/slot connections for workers

}

FtmwViewWidget::~FtmwViewWidget()
{
    for(int i=0; i<d_threadList.size(); i++)
    {
        if(d_threadList.at(i)->isRunning())
        {
            d_threadList[i]->quit();
            d_threadList[i]->wait();
        }
    }

    delete ui;
}

void FtmwViewWidget::prepareForExperiment(const Experiment e)
{
    FtmwConfig config = e.ftmwConfig();
    if(config.type() == BlackChirp::FtmwPeakUp)
        ui->exptLabel->setText(QString("Peak Up Mode"));
    else
        ui->exptLabel->setText(QString("Experiment %1").arg(e.number()));

    if(!ui->exptLabel->isVisible())
        ui->exptLabel->setVisible(true);

    ui->liveFidPlot->prepareForExperiment(e);
    ui->liveFidPlot->setVisible(true);

    ui->fidPlot1->prepareForExperiment(e);
    ui->fidPlot2->prepareForExperiment(e);

//    ui->peakFindWidget->prepareForExperiment(e);

    ui->liveFtPlot->prepareForExperiment(e);
    ui->ftPlot1->prepareForExperiment(e);
    ui->ftPlot2->prepareForExperiment(e);
    ui->mainFtPlot->prepareForExperiment(e);

    d_liveFidList.clear();
    d_currentLiveFid = Fid();
    d_currentFid1 = Fid();
    d_currentFid2 = Fid();
    d_currentLiveFt = Ft();
    d_currentFt1 = Ft();
    d_currentFt2 = Ft();

    d_frame1 = 0;
    d_frame2 = 0;
    d_segment1 = 0;
    d_segment2 = 0;

    if(config.isEnabled())
    {
        p_liveFtw = new FtWorker(d_liveFtwId);
        p_liveFtw->moveToThread(p_liveThread);
        connect(p_liveThread,&QThread::finished,p_liveFtw,&FtWorker::deleteLater);
        connect(p_liveFtw,&FtWorker::fidDone,this,&FtmwViewWidget::fidProcessed);
        connect(p_liveFtw,&FtWorker::ftDone,this,&FtmwViewWidget::ftDone);
        p_liveThread->start();

        d_currentExptNum = e.number();

        ui->verticalLayout->setStretch(0,1);
        ui->liveFidPlot->show();
        ui->liveFtPlot->show();

        if(config.type() == BlackChirp::FtmwPeakUp)
        {

            ///TODO: Implement these elsewhere!
//            connect(ui->rollingAverageSpinbox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwViewWidget::rollingAverageShotsChanged,Qt::UniqueConnection);
//            connect(ui->rollingAverageResetButton,&QPushButton::clicked,this,&FtmwViewWidget::rollingAverageReset,Qt::UniqueConnection);
        }
        if(config.type() == BlackChirp::FtmwLoScan)
            d_mode = BothSB;
        else
            d_mode = Live;
    }
    else
    {        

    }

    d_ftmwConfig = config;

}

void FtmwViewWidget::updateLiveFidList(const FidList fl, int segment)
{
    if(fl.isEmpty())
        return;

    d_liveFidList = fl;

    if(p_liveThread->isRunning())
    {
        d_currentLiveFid = fl.first();
        QMetaObject::invokeMethod(p_liveFtw,"doFT",Q_ARG(Fid,d_currentLiveFid),Q_ARG(FtWorker::FidProcessingSettings,d_currentProcessingSettings));
    }

    if(segment == d_segment1 && d_frame1 < fl.size() && d_frame1 >= 0)
    {
        ///TODO: figure out whether to use snapshots to modify
        /// Then need to reprocess
        d_currentFid1 = fl.at(d_frame1);
    }

    if(segment == d_segment2 && d_frame2 < fl.size() && d_frame2 >= 0)
    {
        ///TODO: figure out whether to use snapshots to modify
        /// Then need to reprocess
        d_currentFid2 = fl.at(d_frame2);
    }

}

void FtmwViewWidget::updateFtmw(const FtmwConfig f)
{
    d_ftmwConfig = f;

    d_currentFid1 = f.singleFid(d_frame1,d_segment1);
    d_currentFid2 = f.singleFid(d_frame2,d_segment2);

    ///TODO: process. Also figure out if main plot needs to be reprocessed.

}

void FtmwViewWidget::fidProcessed(const QVector<QPointF> fidData, int workerId)
{
    FidPlot *plot = nullptr;

    switch(workerId)
    {
    case d_liveFtwId:
        plot = ui->liveFidPlot;
        break;
    case d_plot1FtwId:
        plot = ui->fidPlot1;
        break;
    case d_plot2FtwId:
        plot = ui->fidPlot2;
        break;
    default:
        break;
    }

    if(plot != nullptr)
    {
        if(plot->isVisible())
            plot->receiveProcessedFid(fidData);
    }
}

void FtmwViewWidget::ftDone(const Ft ft, int workerId)
{
    FtPlot *plot = nullptr;

    switch (workerId) {
    case d_liveFtwId:
        plot = ui->liveFtPlot;
        d_currentLiveFt = ft;
        if(d_mode == Live)
            updateMainPlot();
        break;
    case d_plot1FtwId:
        plot = ui->ftPlot1;
        d_currentFt1 = ft;
        if(d_mode == FT1 || d_mode == FT1mFT2 || d_mode == FT2mFT1)
            updateMainPlot();
        break;
    case d_plot2FtwId:
        plot = ui->ftPlot2;
        d_currentFt2 = ft;
        if(d_mode == FT2 || d_mode == FT1mFT2 || d_mode == FT2mFT1)
            updateMainPlot();
        break;
    case d_mainFtwId:
        plot = ui->mainFtPlot;
        break;
    default:
        break;
    }

    if(plot != nullptr)
    {
        if(plot->isVisible())
            plot->newFt(ft);
    }
}

void FtmwViewWidget::updateMainPlot()
{
    switch(d_mode) {
    case Live:
        ui->mainFtPlot->newFt(d_currentLiveFt);
        break;
    case FT1:
        ui->mainFtPlot->newFt(d_currentFt1);
        break;
    case FT2:
        ui->mainFtPlot->newFt(d_currentFt2);
        break;
    case FT1mFT2:
        ///TODO: FT difference
        break;
    case FT2mFT1:
        ///TODO: FT difference
        break;
    case UpperSB:
        break;
    case LowerSB:
        break;
    case BothSB:
        break;
    }
}

void FtmwViewWidget::modeChanged(MainPlotMode newMode)
{
    d_mode = newMode;
    updateMainPlot();
}

void FtmwViewWidget::snapshotTaken()
{
    if(d_currentExptNum < 1)
        return;

//    if(p_snapWidget == nullptr)
//    {
//        p_snapWidget = new FtmwSnapshotWidget(d_currentExptNum,d_path,this);
//        connect(p_snapWidget,&FtmwSnapshotWidget::loadFailed,this,&FtmwViewWidget::snapshotLoadError);
//        connect(p_snapWidget,&FtmwSnapshotWidget::snapListChanged,this,&FtmwViewWidget::snapListUpdate);
//        connect(p_snapWidget,&FtmwSnapshotWidget::refChanged,this,&FtmwViewWidget::snapRefChanged);
//        connect(p_snapWidget,&FtmwSnapshotWidget::diffChanged,this,&FtmwViewWidget::snapDiffChanged);
//        connect(p_snapWidget,&FtmwSnapshotWidget::finalizedList,this,&FtmwViewWidget::finalizedSnapList);
//        connect(p_snapWidget,&FtmwSnapshotWidget::experimentLogMessage,this,&FtmwViewWidget::experimentLogMessage);
//        p_snapWidget->setSelectionEnabled(false);
//        p_snapWidget->setDiffMode(false);
//        p_snapWidget->setFinalizeEnabled(false);
//        if(!ui->controlFrame->isVisible())
//            p_snapWidget->setVisible(false);
//        ui->rightLayout->addWidget(p_snapWidget);
//    }

//    p_snapWidget->setFidList(d_currentFidList);

//    if(p_snapWidget->readSnapshots())
//        ui->snapDiffButton->setEnabled(true);

}

void FtmwViewWidget::experimentComplete()
{
    ui->verticalLayout->setStretch(0,0);
    ui->liveFidPlot->hide();
    ui->liveFtPlot->hide();

    if(p_liveThread->isRunning())
    {
        p_liveThread->quit();
        p_liveThread->wait();

        p_liveFtw = nullptr;
    }

    if(d_mode == Live)
        modeChanged(FT1);
}

void FtmwViewWidget::snapshotLoadError(QString msg)
{
//    p_snapWidget->setEnabled(false);
//    p_snapWidget->deleteLater();
//    p_snapWidget = nullptr;
//    if(ui->snapDiffButton->isChecked())
//        ui->singleFrameButton->setChecked(true);

//    ui->snapDiffButton->setEnabled(false);

//    QMessageBox::warning(this,QString("Snapshot Load Error"),msg,QMessageBox::Ok);
}

void FtmwViewWidget::snapListUpdate()
{
//    if(d_mode == BlackChirp::FtmwViewSnapDiff || ui->snapshotCheckbox->isChecked())
//        updateFtPlot();
}

void FtmwViewWidget::snapRefChanged()
{
//    if(d_mode == BlackChirp::FtmwViewSnapDiff)
//    {
//        d_currentRefFid = p_snapWidget->getRefFid(ui->frameBox->value()-1);
//        updateFtPlot();
//    }
}

void FtmwViewWidget::finalizedSnapList(const FidList l)
{
//    Q_ASSERT(l.size() > 0);
//    d_currentFidList = l;
//    updateShotsLabel(l.first().shots());

//    if(ui->snapshotCheckbox->isChecked())
//    {
//        ui->snapshotCheckbox->blockSignals(true);
//        ui->snapshotCheckbox->setChecked(false);
//        ui->snapshotCheckbox->setEnabled(false);
//        ui->snapshotCheckbox->blockSignals(false);
//    }

//    if(d_mode == BlackChirp::FtmwViewSnapDiff)
//    {
//        ui->singleFrameButton->blockSignals(true);
//        ui->singleFrameButton->setChecked(true);
//        ui->singleFrameButton->blockSignals(false);
//        d_mode = BlackChirp::FtmwViewSingle;
//    }

    emit finalized(d_currentExptNum);
}

