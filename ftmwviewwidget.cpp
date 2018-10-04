#include "ftmwviewwidget.h"

#include <QThread>
#include <QMessageBox>
#include <QMenu>
#include <QToolButton>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QWidgetAction>
#include <QFormLayout>

#include "ftworker.h"
#include "ftmwsnapshotwidget.h"

FtmwViewWidget::FtmwViewWidget(QWidget *parent, QString path) :
    QWidget(parent),
    ui(new Ui::FtmwViewWidget), d_currentExptNum(-1), d_currentSegment(-1), d_mode(Live), d_path(path)
{
    ui->setupUi(this);

    QSettings s;
    s.beginGroup(QString("fidProcessing"));
    double start = s.value(QString("startUs"),-1.0).toDouble();
    double end = s.value(QString("endUs"),-1.0).toDouble();
    int zeroPad = s.value(QString("zeroPad"),0).toInt();
    bool rdc = s.value(QString("removeDC"),false).toBool();
    auto units = static_cast<BlackChirp::FtPlotUnits>(s.value(QString("ftUnits"),BlackChirp::FtPlotuV).toInt());
    double asIg = s.value(QString("autoscaleIgnoreMHz"),0.0).toDouble();
    auto winf = static_cast<BlackChirp::FtWindowFunction>(s.value(QString("windowFunction"),BlackChirp::Boxcar).toInt());
    s.endGroup();

    d_currentProcessingSettings = FtWorker::FidProcessingSettings { start, end, zeroPad, rdc, units, asIg, winf };
    ui->processingWidget->applySettings(d_currentProcessingSettings);
    d_workerIds << d_liveFtwId << d_mainFtwId << d_plot1FtwId << d_plot2FtwId;

    for(int i=0; i<d_workerIds.size(); i++)
    {
        int id = d_workerIds.at(i);
        if(id == d_liveFtwId)
            d_workersStatus.insert(d_liveFtwId, WorkerStatus { nullptr, new QThread(this), false, false} );
        else
        {
            WorkerStatus ws { new FtWorker(id), new QThread(this), false, false};
            connect(ws.thread,&QThread::finished,ws.worker,&FtWorker::deleteLater);
            connect(ws.worker,&FtWorker::ftDone,this,&FtmwViewWidget::ftDone);
            connect(ws.worker,&FtWorker::fidDone,this,&FtmwViewWidget::fidProcessed);
            if(id == d_mainFtwId)
                connect(ws.worker,&FtWorker::ftDiffDone,this,&FtmwViewWidget::ftDiffDone);
            ws.worker->moveToThread(ws.thread);
            ws.thread->start();
            d_workersStatus.insert(id,ws);
        }

        if(id == d_liveFtwId)
            d_plotStatus.insert(id,PlotStatus { ui->liveFidPlot, ui->liveFtPlot, Fid(), Ft(), 0, 0 });
        else if(id == d_plot1FtwId)
            d_plotStatus.insert(id,PlotStatus { ui->fidPlot1, ui->ftPlot1, Fid(), Ft(), 0, 0 });
        else if(id == d_plot2FtwId)
            d_plotStatus.insert(id,PlotStatus { ui->fidPlot2, ui->ftPlot2, Fid(), Ft(), 0, 0 });
        //don't need to add one of these for the main plot; it's special

    }

    ui->snapshotWidget1->hide();
    ui->snapshotWidget2->hide();

    connect(ui->processingWidget,&FtmwProcessingWidget::settingsUpdated,this,&FtmwViewWidget::updateProcessingSettings);
    connect(ui->processingMenu,&QMenu::aboutToHide,this,&FtmwViewWidget::storeProcessingSettings);

    connect(ui->plot1ConfigWidget,&FtmwPlotConfigWidget::frameChanged,this,[=](int v){
        changeFrame(d_plot1FtwId,v);
    });
    connect(ui->plot1ConfigWidget,&FtmwPlotConfigWidget::segmentChanged,this,[=](int v){
        changeSegment(d_plot1FtwId,v);
    });
    connect(ui->plot2ConfigWidget,&FtmwPlotConfigWidget::frameChanged,this,[=](int v){
        changeFrame(d_plot2FtwId,v);
    });
    connect(ui->plot2ConfigWidget,&FtmwPlotConfigWidget::segmentChanged,this,[=](int v){
        changeSegment(d_plot2FtwId,v);
    });

    connect(ui->liveAction,&QAction::triggered,this,[=]() { modeChanged(Live); });
    connect(ui->ft1Action,&QAction::triggered,this,[=]() { modeChanged(FT1); });
    connect(ui->ft2Action,&QAction::triggered,this,[=]() { modeChanged(FT2); });
    connect(ui->ft12DiffAction,&QAction::triggered,this,[=]() { modeChanged(FT1mFT2); });
    connect(ui->ft21DiffAction,&QAction::triggered,this,[=]() { modeChanged(FT2mFT1); });
    connect(ui->usAction,&QAction::triggered,this,[=]() { modeChanged(UpperSB); });
    connect(ui->lsAction,&QAction::triggered,this,[=]() { modeChanged(LowerSB); });
    connect(ui->bsAction,&QAction::triggered,this,[=]() { modeChanged(BothSB); });

}

FtmwViewWidget::~FtmwViewWidget()
{
    for(auto it=d_workersStatus.begin(); it != d_workersStatus.end(); it++)
    {
        it.value().thread->quit();
        it.value().thread->wait();
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
    ui->processingWidget->prepareForExperient(e);
    ui->ftPlot1->prepareForExperiment(e);
    ui->ftPlot2->prepareForExperiment(e);
    ui->mainFtPlot->prepareForExperiment(e);
    ui->plot1ConfigWidget->prepareForExperiment(e);
    ui->plot2ConfigWidget->prepareForExperiment(e);

    d_liveFidList.clear();
    d_currentSegment = 0;
    for(auto it = d_plotStatus.begin(); it != d_plotStatus.end(); it++)
    {
        it.value().fid = Fid();
        it.value().ft = Ft();
        it.value().frame = 0;
        it.value().segment = 0;
    }

    if(config.isEnabled())
    {
        auto ws = d_workersStatus.value(d_liveFtwId);
        ws.worker = new FtWorker(d_liveFtwId);
        ws.worker->moveToThread(ws.thread);
        connect(ws.thread,&QThread::finished,ws.worker,&FtWorker::deleteLater);
        connect(ws.worker,&FtWorker::fidDone,this,&FtmwViewWidget::fidProcessed);
        connect(ws.worker,&FtWorker::ftDone,this,&FtmwViewWidget::ftDone);
        ws.busy = false;
        ws.reprocessWhenDone = false;
        ws.thread->start();
        d_workersStatus.insert(d_liveFtwId,ws);

        d_currentExptNum = e.number();

        ui->verticalLayout->setStretch(0,1);
        ui->liveFidPlot->show();
        ui->liveFtPlot->show();

        ui->liveAction->setEnabled(true);

        if(config.type() == BlackChirp::FtmwPeakUp)
        {

            ///TODO: Implement these elsewhere!
//            connect(ui->rollingAverageSpinbox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwViewWidget::rollingAverageShotsChanged,Qt::UniqueConnection);
//            connect(ui->rollingAverageResetButton,&QPushButton::clicked,this,&FtmwViewWidget::rollingAverageReset,Qt::UniqueConnection);
        }
        if(config.type() == BlackChirp::FtmwLoScan)
        {
//            d_mode = BothSB;
            ui->bsAction->setEnabled(true);
            ui->usAction->setEnabled(true);
            ui->lsAction->setEnabled(true);
            ui->bsAction->trigger();
        }
        else
        {
            ui->liveAction->trigger();
            ui->bsAction->setEnabled(false);
            ui->usAction->setEnabled(false);
            ui->lsAction->setEnabled(false);
        }
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
    d_currentSegment = segment;

    for(auto it = d_plotStatus.begin(); it != d_plotStatus.end(); it++)
    {
        if(d_workersStatus.value(it.key()).thread->isRunning())
        {
            Fid f = fl.first();
            if(it.key() != d_liveFtwId)
            {
                if(segment == it.value().segment && it.value().frame < fl.size())
                {
                    f = fl.at(it.value().frame);
                    it.value().fid = f;
                    process(it.key(),f);
                }
            }
            else
            {
                it.value().fid = f;
                process(it.key(),f);
            }

        }
    }
}

void FtmwViewWidget::updateFtmw(const FtmwConfig f)
{
    d_ftmwConfig = f;

    for(auto it = d_plotStatus.begin(); it != d_plotStatus.end(); it++)
    {
        if(it.key() == d_liveFtwId)
            continue;

        it.value().fid = f.singleFid(it.value().frame,it.value().segment);
    }

    reprocess(QList<int>{ d_liveFtwId });

}

void FtmwViewWidget::updateProcessingSettings(FtWorker::FidProcessingSettings s)
{
    d_currentProcessingSettings = s;
    QList<int> ignore;
    switch(d_mode)
    {
    case UpperSB:
    case LowerSB:
    case BothSB:
        ignore << d_mainFtwId;
    default:
        break;
    }

    if(!ui->liveFidPlot->isHidden())
    {
        ui->liveFidPlot->setFtStart(s.startUs);
        ui->liveFidPlot->setFtEnd(s.endUs);
    }
    ui->fidPlot1->setFtStart(s.startUs);
    ui->fidPlot1->setFtEnd(s.endUs);
    ui->fidPlot2->setFtStart(s.startUs);
    ui->fidPlot2->setFtEnd(s.endUs);

    reprocess(ignore);
}

void FtmwViewWidget::storeProcessingSettings()
{
    QSettings s;
    s.beginGroup(QString("fidProcessing"));
    s.setValue(QString("startUs"),d_currentProcessingSettings.startUs);
    s.setValue(QString("endUs"),d_currentProcessingSettings.endUs);
    s.setValue(QString("autoscaleIgnoreMHz"),d_currentProcessingSettings.autoScaleIgnoreMHz);
    s.setValue(QString("zeroPad"),d_currentProcessingSettings.zeroPadFactor);
    s.setValue(QString("removeDC"),d_currentProcessingSettings.removeDC);
    s.setValue(QString("windowFunction"),static_cast<int>(d_currentProcessingSettings.windowFunction));
    s.setValue(QString("ftUnits"),static_cast<int>(d_currentProcessingSettings.units));
    s.endGroup();
    s.sync();

    reprocessAll();
}

void FtmwViewWidget::changeFrame(int id, int frameNum)
{
    if(d_plotStatus.contains(id))
    {
        d_plotStatus[id].frame = frameNum;
        updateFid(id);
    }
}

void FtmwViewWidget::changeSegment(int id, int segmentNum)
{
    if(d_plotStatus.contains(id))
    {
        d_plotStatus[id].segment = segmentNum;
        updateFid(id);
    }
}

void FtmwViewWidget::fidProcessed(const QVector<QPointF> fidData, int workerId)
{
    if(d_plotStatus.contains(workerId))
    {
        if(!d_plotStatus.value(workerId).fidPlot->isHidden())
            d_plotStatus[workerId].fidPlot->receiveProcessedFid(fidData);
    }
}

void FtmwViewWidget::ftDone(const Ft ft, int workerId)
{
    if(d_plotStatus.contains(workerId))
    {
        if(!d_plotStatus.value(workerId).ftPlot->isHidden())
        {
            d_plotStatus[workerId].ft = ft;
            d_plotStatus[workerId].ftPlot->configureUnits(d_currentProcessingSettings.units);
            d_plotStatus[workerId].ftPlot->newFt(ft);
        }

        switch(d_mode) {
        case Live:
        case FT1:
        case FT2:
        case FT1mFT2:
        case FT2mFT1:
            updateMainPlot();
            break;
        default:
            break;
        }
    }
    else
    {
        //this is the main plot
        ui->mainFtPlot->newFt(ft);
    }

    d_workersStatus[workerId].busy = false;
    if(d_workersStatus.value(workerId).reprocessWhenDone)
        process(workerId,d_plotStatus.value(workerId).fid);
}

void FtmwViewWidget::ftDiffDone(const Ft ft)
{
    ui->mainFtPlot->newFt(ft);

    d_workersStatus[d_mainFtwId].busy = false;
    if(d_workersStatus.value(d_mainFtwId).reprocessWhenDone)
    {
        //need to set the reprocess flag here in case mode has changed since job started
        d_workersStatus[d_mainFtwId].reprocessWhenDone = false;
        updateMainPlot();
    }
}

void FtmwViewWidget::updateMainPlot()
{
    ui->mainFtPlot->configureUnits(d_currentProcessingSettings.units);

    switch(d_mode) {
    case Live:
        ui->mainFtPlot->newFt(d_plotStatus.value(d_liveFtwId).ft);
        break;
    case FT1:
        ui->mainFtPlot->newFt(d_plotStatus.value(d_plot1FtwId).ft);
        break;
    case FT2:
        ui->mainFtPlot->newFt(d_plotStatus.value(d_plot2FtwId).ft);
        break;
    case FT1mFT2:
        processDiff(d_plotStatus.value(d_plot1FtwId).fid,d_plotStatus.value(d_plot2FtwId).fid);
        break;
    case FT2mFT1:
        processDiff(d_plotStatus.value(d_plot2FtwId).fid,d_plotStatus.value(d_plot1FtwId).fid);
        break;
    case UpperSB:
        break;
    case LowerSB:
        break;
    case BothSB:
        break;
    }
}

void FtmwViewWidget::reprocessAll()
{
    reprocess();
}

void FtmwViewWidget::reprocess(const QList<int> ignore)
{
    for(auto it=d_workersStatus.constBegin(); it != d_workersStatus.constEnd(); it++)
    {
        if(!ignore.contains(it.key()))
        {
            if(it.key() == d_mainFtwId)
                updateMainPlot();
            else
                process(it.key(),d_plotStatus.value(it.key()).fid);
        }
    }
}

void FtmwViewWidget::process(int id, const Fid f)
{
//    if(f.isEmpty())
//        return;

    auto ws = d_workersStatus.value(id);
    if(ws.thread->isRunning() && ws.worker != nullptr)
    {
        if(ws.busy)
            d_workersStatus[id].reprocessWhenDone = true;
        else
        {
            d_workersStatus[id].busy = true;
            d_workersStatus[id].reprocessWhenDone = false;
            QMetaObject::invokeMethod(ws.worker,"doFT",Q_ARG(Fid,f),Q_ARG(FtWorker::FidProcessingSettings,d_currentProcessingSettings));
        }
    }
}

void FtmwViewWidget::processDiff(const Fid f1, const Fid f2)
{
    if(f1.isEmpty() || f2.isEmpty())
        return;

    auto ws = d_workersStatus.value(d_mainFtwId);
    if(ws.busy)
        d_workersStatus[d_mainFtwId].reprocessWhenDone = true;
    else
    {
         d_workersStatus[d_mainFtwId].busy = true;
         d_workersStatus[d_mainFtwId].reprocessWhenDone = false;
         QMetaObject::invokeMethod(ws.worker,"doFtDiff",Q_ARG(Fid,f1),Q_ARG(Fid,f2),Q_ARG(FtWorker::FidProcessingSettings,d_currentProcessingSettings));
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

void FtmwViewWidget::experimentComplete(const Experiment e)
{
    if(e.ftmwConfig().isEnabled())
    {
        d_currentSegment = -1;

        ui->verticalLayout->setStretch(0,0);
        ui->liveFidPlot->hide();
        ui->liveFtPlot->hide();


        if(d_workersStatus.value(d_liveFtwId).thread->isRunning())
        {
            d_workersStatus[d_liveFtwId].thread->quit();
            d_workersStatus[d_liveFtwId].thread->wait();

            d_workersStatus[d_liveFtwId].worker = nullptr;
        }

        if(d_mode == Live)
            ui->ft1Action->trigger();

        ui->liveAction->setEnabled(false);

        updateFtmw(e.ftmwConfig());
    }
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

void FtmwViewWidget::updateFid(int id)
{
    int seg = d_plotStatus.value(id).segment;
    int frame = d_plotStatus.value(id).frame;

    if(seg == d_currentSegment)
    {
        if(frame >= 0 && frame < d_liveFidList.size())
            d_plotStatus[id].fid = d_liveFidList.at(frame);
    }
    else
        d_plotStatus[id].fid = d_ftmwConfig.singleFid(frame,seg);

    process(id, d_plotStatus.value(id).fid);

}

