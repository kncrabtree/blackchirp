#include <gui/widget/ftmwviewwidget.h>

#include <QThread>
#include <QMessageBox>
#include <QMenu>
#include <QToolButton>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QWidgetAction>
#include <QFormLayout>

#include <data/analysis/ftworker.h>
#include <gui/widget/peakfindwidget.h>

FtmwViewWidget::FtmwViewWidget(QWidget *parent, QString path) :
    QWidget(parent),
    ui(new Ui::FtmwViewWidget), d_currentExptNum(-1), d_currentSegment(-1), d_mode(Live), d_path(path)
{
    ui->setupUi(this,d_path);

    p_pfw = nullptr;

    d_currentProcessingSettings = ui->processingWidget->getSettings();

    d_workerIds << d_liveId << d_mainId << d_plot1Id << d_plot2Id;

    for(int i=0; i<d_workerIds.size(); i++)
    {
        int id = d_workerIds.at(i);
        if(id == d_liveId)
            d_workersStatus.insert(d_liveId, WorkerStatus { nullptr, new QThread(this), false, false} );
        else
        {
            WorkerStatus ws { new FtWorker(id), new QThread(this), false, false};
            connect(ws.thread,&QThread::finished,ws.worker,&FtWorker::deleteLater);
            connect(ws.worker,&FtWorker::ftDone,this,&FtmwViewWidget::ftDone);
            connect(ws.worker,&FtWorker::fidDone,this,&FtmwViewWidget::fidProcessed);
            if(id == d_mainId)
                connect(ws.worker,&FtWorker::ftDiffDone,this,&FtmwViewWidget::ftDiffDone);
            ws.worker->moveToThread(ws.thread);
            ws.thread->start();
            d_workersStatus.insert(id,ws);
        }

        if(id == d_liveId)
            d_plotStatus.insert(id,PlotStatus { ui->liveFidPlot, ui->liveFtPlot, Fid(), Ft(), 0, 0, false });
        else if(id == d_plot1Id)
            d_plotStatus.insert(id,PlotStatus { ui->fidPlot1, ui->ftPlot1, Fid(), Ft(), 0, 0, false });
        else if(id == d_plot2Id)
            d_plotStatus.insert(id,PlotStatus { ui->fidPlot2, ui->ftPlot2, Fid(), Ft(), 0, 0, false });
        //don't need to add one of these for the main plot; it's special

    }

    for(auto it = d_plotStatus.begin(); it != d_plotStatus.end(); it++)
    {
        it.value().fidPlot->blockSignals(true);
        it.value().fidPlot->setFtStart(d_currentProcessingSettings.startUs);
        it.value().fidPlot->setFtEnd(d_currentProcessingSettings.endUs);
        it.value().fidPlot->blockSignals(false);
    }

    connect(ui->processingWidget,&FtmwProcessingWidget::settingsUpdated,this,&FtmwViewWidget::updateProcessingSettings);
    connect(ui->processingMenu,&QMenu::aboutToHide,[this](){this->reprocess();});

    connect(ui->plot1ConfigWidget,&FtmwPlotConfigWidget::frameChanged,this,&FtmwViewWidget::changeFrame);
    connect(ui->plot1ConfigWidget,&FtmwPlotConfigWidget::segmentChanged,this,&FtmwViewWidget::changeSegment);
    connect(ui->plot1ConfigWidget,&FtmwPlotConfigWidget::snapshotsProcessed,this,&FtmwViewWidget::snapshotsProcessed);
    connect(ui->plot1ConfigWidget,&FtmwPlotConfigWidget::snapshotsFinalized,this,&FtmwViewWidget::snapshotsFinalized);
    connect(ui->plot2ConfigWidget,&FtmwPlotConfigWidget::frameChanged,this,&FtmwViewWidget::changeFrame);
    connect(ui->plot2ConfigWidget,&FtmwPlotConfigWidget::segmentChanged,this,&FtmwViewWidget::changeSegment);
    connect(ui->plot2ConfigWidget,&FtmwPlotConfigWidget::snapshotsProcessed,this,&FtmwViewWidget::snapshotsProcessed);
    connect(ui->plot2ConfigWidget,&FtmwPlotConfigWidget::snapshotsFinalized,this,&FtmwViewWidget::snapshotsFinalized);

    connect(ui->liveAction,&QAction::triggered,this,[=]() { modeChanged(Live); });
    connect(ui->ft1Action,&QAction::triggered,this,[=]() { modeChanged(FT1); });
    connect(ui->ft2Action,&QAction::triggered,this,[=]() { modeChanged(FT2); });
    connect(ui->ft12DiffAction,&QAction::triggered,this,[=]() { modeChanged(FT1mFT2); });
    connect(ui->ft21DiffAction,&QAction::triggered,this,[=]() { modeChanged(FT2mFT1); });
    connect(ui->usAction,&QAction::triggered,this,[=]() { modeChanged(UpperSB); });
    connect(ui->lsAction,&QAction::triggered,this,[=]() { modeChanged(LowerSB); });
    connect(ui->bsAction,&QAction::triggered,this,[=]() { modeChanged(BothSB); });

    connect(ui->peakFindAction,&QAction::triggered,this,&FtmwViewWidget::launchPeakFinder);

    connect(ui->minFtSegBox,static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),this,&FtmwViewWidget::updateSidebandFreqs);
    connect(ui->maxFtSegBox,static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),this,&FtmwViewWidget::updateSidebandFreqs);

    connect(ui->averagesSpinbox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwViewWidget::rollingAverageShotsChanged,Qt::UniqueConnection);
    connect(ui->resetAveragesButton,&QPushButton::clicked,this,&FtmwViewWidget::rollingAverageReset,Qt::UniqueConnection);

}

FtmwViewWidget::~FtmwViewWidget()
{
    for(auto it=d_workersStatus.begin(); it != d_workersStatus.end(); it++)
    {
        it.value().thread->quit();
        it.value().thread->wait();
    }

    if(p_pfw != nullptr)
        p_pfw->close();

    delete ui;
}

void FtmwViewWidget::prepareForExperiment(const Experiment e)
{
    FtmwConfig config = e.ftmwConfig();
    p_fidStorage = config.storage();
    if(config.type() == BlackChirp::FtmwPeakUp)
        ui->exptLabel->setText(QString("Peak Up Mode"));
    else
        ui->exptLabel->setText(QString("Experiment %1").arg(e.d_number));


    if(p_pfw != nullptr)
    {
        p_pfw->close();
        p_pfw = nullptr;
    }

    if(!ui->exptLabel->isVisible())
        ui->exptLabel->setVisible(true);

    ui->shotsLabel->setText(d_shotsString.arg(0));

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
        auto ws = d_workersStatus.value(d_liveId);
        ws.worker = new FtWorker(d_liveId);
        ws.worker->moveToThread(ws.thread);
        connect(ws.thread,&QThread::finished,ws.worker,&FtWorker::deleteLater);
        connect(ws.worker,&FtWorker::fidDone,this,&FtmwViewWidget::fidProcessed);
        connect(ws.worker,&FtWorker::ftDone,this,&FtmwViewWidget::ftDone);
        ws.busy = false;
        ws.reprocessWhenDone = false;
        ws.thread->start();
        d_workersStatus.insert(d_liveId,ws);

        d_currentExptNum = e.d_number;

        ui->verticalLayout->setStretch(0,1);
        ui->liveFidPlot->show();
        ui->liveFtPlot->show();

        ui->liveAction->setEnabled(true);

        ui->averagesSpinbox->blockSignals(true);
        ui->averagesSpinbox->setValue(config.targetShots());
        ui->averagesSpinbox->blockSignals(false);

        ui->resetAveragesButton->setEnabled(config.type() == BlackChirp::FtmwPeakUp);
        ui->averagesSpinbox->setEnabled(config.type() == BlackChirp::FtmwPeakUp);

        auto chirpOffsetRange = config.rfConfig().calculateChirpAbsOffsetRange();
        if(chirpOffsetRange.first < 0.0)
            chirpOffsetRange.first = 0.0;
        if(chirpOffsetRange.second < 0.0)
            chirpOffsetRange.second = config.ftNyquistMHz();

        ui->minFtSegBox->blockSignals(true);
        ui->minFtSegBox->setRange(0.0,config.ftNyquistMHz());
        ui->minFtSegBox->setValue(chirpOffsetRange.first);
        ui->minFtSegBox->blockSignals(false);

        ui->maxFtSegBox->blockSignals(true);
        ui->maxFtSegBox->setRange(0.0,config.ftNyquistMHz());
        ui->maxFtSegBox->setValue(chirpOffsetRange.second);
        ui->maxFtSegBox->blockSignals(false);

        if(config.type() == BlackChirp::FtmwLoScan)
        {
//            d_mode = BothSB;
            ui->bsAction->setEnabled(true);
            ui->usAction->setEnabled(true);
            ui->lsAction->setEnabled(true);
            ui->minFtSegBox->setEnabled(true);
            ui->maxFtSegBox->setEnabled(true);
            ui->bsAction->trigger();
        }
        else
        {
            ui->liveAction->trigger();
            ui->bsAction->setEnabled(false);
            ui->usAction->setEnabled(false);
            ui->lsAction->setEnabled(false);
            ui->minFtSegBox->setEnabled(false);
            ui->maxFtSegBox->setEnabled(false);
        }
    }
    else
    {        
        ui->resetAveragesButton->setEnabled(false);
        ui->averagesSpinbox->setEnabled(false);
    }

    ui->peakFindAction->setEnabled(false);
    d_ftmwConfig = config;
    d_snap1Config = config;
    d_snap2Config = config;

}

void FtmwViewWidget::updateLiveFidList(const FtmwConfig c, int segment)
{
    auto fl = p_fidStorage->getCurrentFidList();
    d_currentSegment = p_fidStorage->getCurrentIndex();

//    d_ftmwConfig = c;
    d_currentSegment = segment;

    ui->shotsLabel->setText(d_shotsString.arg(p_fidStorage->currentSegmentShots()));


    for(auto it = d_plotStatus.begin(); it != d_plotStatus.end(); it++)
    {
        if(d_workersStatus.value(it.key()).thread->isRunning())
        {
            Fid f = fl.constFirst();
            if(it.key() != d_liveId)
            {
                if(segment == it.value().segment && it.value().frame < fl.size())
                {
                    bool processFid = true;
                    if(it.key() == d_plot1Id)
                    {
//                        ui->plot1ConfigWidget->processFtmwConfig(c);
                        if(ui->plot1ConfigWidget->isSnapshotActive())
                            processFid = false;
                    }
                    else if(it.key() == d_plot2Id)
                    {
//                        ui->plot2ConfigWidget->processFtmwConfig(c);
                        if(ui->plot2ConfigWidget->isSnapshotActive())
                            processFid = false;
                    }

                    if(processFid)
                    {
                        f = fl.at(it.value().frame);
                        it.value().fid = f;
                        process(it.key(),f);
                    }
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
    QList<int> ignore{ d_liveId };

    for(auto it = d_plotStatus.begin(); it != d_plotStatus.end(); it++)
    {
        if(it.key() == d_liveId)
            continue;

        if(it.key() == d_plot1Id && ui->plot1ConfigWidget->isSnapshotActive())
        {
            ignore << it.key();
            ui->plot1ConfigWidget->processFtmwConfig(f);
        }
        else if(it.key() == d_plot2Id && ui->plot2ConfigWidget->isSnapshotActive())
        {
            ignore << it.key();
            ui->plot2ConfigWidget->processFtmwConfig(f);
        }
        else
            it.value().fid = p_fidStorage->getFidList(it.value().segment).at(it.value().frame);
    }

    reprocess(ignore);

}

void FtmwViewWidget::updateProcessingSettings(FtWorker::FidProcessingSettings s)
{
    //skip main plot because it will be updated when menu is closed
    d_currentProcessingSettings = s;
    QList<int> ignore;
    switch(d_mode)
    {
    case UpperSB:
    case LowerSB:
    case BothSB:
        ignore << d_mainId;
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

        d_plotStatus[workerId].fidPlot->setCursor(Qt::CrossCursor);
        d_plotStatus[workerId].ftPlot->setCursor(Qt::CrossCursor);

        switch(d_mode) {
        case Live:
        case FT1:
        case FT2:
        case FT1mFT2:
        case FT2mFT1:
            updateMainPlot();
            break;
        default:
            if(workerId == d_plot1Id && ui->plot1ConfigWidget->isSnapshotActive() && ui->mainPlotFollowSpinBox->value() == 1)
                updateMainPlot();
            else if(workerId == d_plot2Id && ui->plot2ConfigWidget->isSnapshotActive() && ui->mainPlotFollowSpinBox->value() == 2)
                updateMainPlot();
            break;
        }
    }
    else
    {
        //this is the main plot
        ui->mainFtPlot->newFt(ft);
        ui->mainFtPlot->canvas()->setCursor(QCursor(Qt::CrossCursor));
        if(p_pfw != nullptr)
            p_pfw->newFt(ft);
    }

    d_workersStatus[workerId].busy = false;
    if(d_workersStatus.value(workerId).reprocessWhenDone)
    {
        if(workerId == d_mainId)
            updateMainPlot();
        else
            process(workerId,d_plotStatus.value(workerId).fid);
    }
}

void FtmwViewWidget::ftDiffDone(const Ft ft)
{
    ui->mainFtPlot->newFt(ft);
    ui->mainFtPlot->canvas()->setCursor(QCursor(Qt::CrossCursor));

    d_workersStatus[d_mainId].busy = false;
    if(d_workersStatus.value(d_mainId).reprocessWhenDone)
    {
        //need to set the reprocess flag here in case mode has changed since job started
        d_workersStatus[d_mainId].reprocessWhenDone = false;
        updateMainPlot();
    }
}

void FtmwViewWidget::updateMainPlot()
{
    ui->mainFtPlot->configureUnits(d_currentProcessingSettings.units);
    if(!ui->mainFtPlot->currentFt().isEmpty())
        ui->peakFindAction->setEnabled(true);

    switch(d_mode) {
    case Live:
        ui->mainFtPlot->newFt(d_plotStatus.value(d_liveId).ft);
        if(p_pfw != nullptr)
            p_pfw->newFt(d_plotStatus.value(d_liveId).ft);
        break;
    case FT1:
        ui->mainFtPlot->newFt(d_plotStatus.value(d_plot1Id).ft);
        if(p_pfw != nullptr)
            p_pfw->newFt(d_plotStatus.value(d_plot1Id).ft);
        break;
    case FT2:
        ui->mainFtPlot->newFt(d_plotStatus.value(d_plot2Id).ft);
        if(p_pfw != nullptr)
            p_pfw->newFt(d_plotStatus.value(d_plot2Id).ft);
        break;
    case FT1mFT2:
        processDiff(d_plotStatus.value(d_plot1Id).fid,d_plotStatus.value(d_plot2Id).fid);
        break;
    case FT2mFT1:
        processDiff(d_plotStatus.value(d_plot2Id).fid,d_plotStatus.value(d_plot1Id).fid);
        break;
    case UpperSB:
        processSideband(BlackChirp::UpperSideband);
        break;
    case LowerSB:
        processSideband(BlackChirp::LowerSideband);
        break;
    case BothSB:
        processBothSidebands();
        break;
    }
}

void FtmwViewWidget::reprocess(const QList<int> ignore)
{
    for(auto it=d_workersStatus.constBegin(); it != d_workersStatus.constEnd(); it++)
    {
        if(!ignore.contains(it.key()))
        {
            if(it.key() == d_mainId)
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
            d_plotStatus[id].fidPlot->setCursor(Qt::BusyCursor);
            d_plotStatus[id].ftPlot->setCursor(Qt::BusyCursor);
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

    auto ws = d_workersStatus.value(d_mainId);
    if(ws.busy)
        d_workersStatus[d_mainId].reprocessWhenDone = true;
    else
    {
        ui->mainFtPlot->canvas()->setCursor(QCursor(Qt::BusyCursor));
        d_workersStatus[d_mainId].busy = true;
        d_workersStatus[d_mainId].reprocessWhenDone = false;
        QMetaObject::invokeMethod(ws.worker,"doFtDiff",Q_ARG(Fid,f1),Q_ARG(Fid,f2),Q_ARG(FtWorker::FidProcessingSettings,d_currentProcessingSettings));
    }
}

void FtmwViewWidget::processSideband(BlackChirp::Sideband sb)
{
    ///TODO: Make this work with the shared storage
    auto ws = d_workersStatus.value(d_mainId);
    if(ws.busy)
        d_workersStatus[d_mainId].reprocessWhenDone = true;
    else
    {
        FidList fl;
        FtmwConfig c = d_ftmwConfig;
        int id = d_plot1Id;
        if(ui->mainPlotFollowSpinBox->value() == 2)
            id = d_plot2Id;

        if(ui->plot1ConfigWidget->isSnapshotActive() && ui->mainPlotFollowSpinBox->value() == 1)
            c = d_snap1Config;
        else if(ui->plot2ConfigWidget->isSnapshotActive() && ui->mainPlotFollowSpinBox->value() == 2)
            c = d_snap2Config;

//        for(int i=0; i<c.multiFidList().size(); i++)
//            fl << c.singleFid(d_plotStatus.value(id).frame,i);

        if(!fl.isEmpty())
        {
            ui->mainFtPlot->canvas()->setCursor(QCursor(Qt::BusyCursor));
            d_workersStatus[d_mainId].busy = true;
            d_workersStatus[d_mainId].reprocessWhenDone = false;
            double minF = ui->minFtSegBox->value();
            double maxF = ui->maxFtSegBox->value();

            QMetaObject::invokeMethod(ws.worker,"processSideband",Q_ARG(FidList,fl),Q_ARG(FtWorker::FidProcessingSettings,d_currentProcessingSettings),Q_ARG(BlackChirp::Sideband,sb),Q_ARG(double,minF),Q_ARG(double,maxF));
        }
    }
}

void FtmwViewWidget::processBothSidebands()
{
    ///TODO: Make this work with the shared storage
    auto ws = d_workersStatus.value(d_mainId);
    if(ws.busy)
        d_workersStatus[d_mainId].reprocessWhenDone = true;
    else
    {
        FidList fl;
        FtmwConfig c = d_ftmwConfig;
        int id = d_plot1Id;
        if(ui->mainPlotFollowSpinBox->value() == 2)
            id = d_plot2Id;

        if(ui->plot1ConfigWidget->isSnapshotActive() && ui->mainPlotFollowSpinBox->value() == 1)
            c = d_snap1Config;
        else if(ui->plot2ConfigWidget->isSnapshotActive() && ui->mainPlotFollowSpinBox->value() == 2)
            c = d_snap2Config;

//        for(int i=0; i<c.multiFidList().size(); i++)
//            fl << c.singleFid(d_plotStatus.value(id).frame,i);

        if(!fl.isEmpty())
        {
            ui->mainFtPlot->canvas()->setCursor(QCursor(Qt::BusyCursor));
            d_workersStatus[d_mainId].busy = true;
            d_workersStatus[d_mainId].reprocessWhenDone = false;
            double minF = ui->minFtSegBox->value();
            double maxF = ui->maxFtSegBox->value();

            QMetaObject::invokeMethod(ws.worker,"processBothSidebands",Q_ARG(FidList,fl),Q_ARG(FtWorker::FidProcessingSettings,d_currentProcessingSettings),Q_ARG(double,minF),Q_ARG(double,maxF));
        }
    }
}

void FtmwViewWidget::updateSidebandFreqs()
{
    if(d_mode == BothSB || d_mode == UpperSB || d_mode == LowerSB)
        reprocess(QList<int>{d_liveId,d_plot1Id,d_plot2Id});
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

    ui->plot1ConfigWidget->snapshotTaken();
    ui->plot2ConfigWidget->snapshotTaken();

}

void FtmwViewWidget::snapshotsProcessed(int id, const FtmwConfig c)
{
    if(id != d_plot1Id && id != d_plot2Id)
        return;

    if(id == d_plot1Id)
    {
        d_snap1Config = c;
        if(ui->plot1ConfigWidget->isSnapshotActive())
            updateFid(id);
    }
    else
    {
        d_snap2Config = c;
        if(ui->plot2ConfigWidget->isSnapshotActive())
            updateFid(id);
    }

}

void FtmwViewWidget::snapshotsFinalized(const FtmwConfig out)
{
    d_ftmwConfig = out;

    Experiment e(d_currentExptNum,d_path);
    qint64 oldNum = e.ftmwConfig().completedShots();
    e.finalizeFtmwSnapshots(out);
    emit experimentLogMessage(e.d_number,QString("Finalized snapshots. Old completed shots: %1. New completed shots: %2").arg(oldNum).arg(e.ftmwConfig().completedShots()));

    ui->shotsLabel->setText(d_shotsString.arg(e.ftmwConfig().completedShots()));


    reprocess();
    emit finalized(d_currentExptNum);

    snapshotsFinalizedUpdateUi(d_currentExptNum);
}

void FtmwViewWidget::snapshotsFinalizedUpdateUi(int num)
{
    if(num == d_currentExptNum)
    {
        ui->plot1ConfigWidget->clearAll();
        ui->plot2ConfigWidget->clearAll();
    }
}

void FtmwViewWidget::experimentComplete(const Experiment e)
{
    ui->plot1ConfigWidget->experimentComplete(e);
    ui->plot2ConfigWidget->experimentComplete(e);

    if(e.ftmwConfig().isEnabled())
    {
        d_currentSegment = -1;

        ui->verticalLayout->setStretch(0,0);
        ui->liveFidPlot->hide();
        ui->liveFtPlot->hide();


        if(d_workersStatus.value(d_liveId).thread->isRunning())
        {
            d_workersStatus[d_liveId].thread->quit();
            d_workersStatus[d_liveId].thread->wait();

            d_workersStatus[d_liveId].worker = nullptr;
        }

        if(d_mode == Live)
            ui->ft1Action->trigger();

        ui->liveAction->setEnabled(false);

        updateFtmw(e.ftmwConfig());
    }

    ui->shotsLabel->setText(d_shotsString.arg(e.ftmwConfig().completedShots()));
}

void FtmwViewWidget::launchPeakFinder()
{
    if(p_pfw != nullptr)
    {
        p_pfw->activateWindow();
        p_pfw->raise();
        return;
    }

    p_pfw = new PeakFindWidget(ui->mainFtPlot->currentFt());
    if(d_currentExptNum > 0)
        p_pfw->setWindowTitle(QString("Peak List: Experiment %1").arg(d_currentExptNum));
    else
        p_pfw->setWindowTitle(QString("Peak List"));

    p_pfw->setAttribute(Qt::WA_DeleteOnClose);

    connect(d_workersStatus.value(d_mainId).worker,&FtWorker::ftDone,p_pfw,&PeakFindWidget::newFt);
    connect(p_pfw,&PeakFindWidget::peakList,ui->mainFtPlot,&FtPlot::newPeakList);
    connect(p_pfw,&PeakFindWidget::destroyed,[=](){
        p_pfw = nullptr;
    });

    p_pfw->show();
    p_pfw->activateWindow();
    p_pfw->raise();

}

void FtmwViewWidget::updateFid(int id)
{
    int seg = d_plotStatus.value(id).segment;
    int frame = d_plotStatus.value(id).frame;

    bool snap = false;
    FtmwConfig c = d_ftmwConfig;
    if(id == d_plot1Id)
    {
        snap = ui->plot1ConfigWidget->isSnapshotActive();
        if(snap)
            c = d_snap1Config;
    }
    else if(id == d_plot2Id)
    {
        snap = ui->plot2ConfigWidget->isSnapshotActive();
        if(snap)
            c = d_snap2Config;
    }

    if(seg == d_currentSegment && !snap)
    {
        auto fl = p_fidStorage->getCurrentFidList();
        if(frame >= 0 && frame < fl.size())
            d_plotStatus[id].fid = fl.at(frame);
    }
    else
        d_plotStatus[id].fid = p_fidStorage->getFidList(seg).at(frame);

    process(id, d_plotStatus.value(id).fid);

}

