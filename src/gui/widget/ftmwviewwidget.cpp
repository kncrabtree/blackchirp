#include <gui/widget/ftmwviewwidget.h>

#include <QThread>
#include <QMessageBox>
#include <QMenu>
#include <QToolButton>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QWidgetAction>
#include <QFormLayout>
#include <QtConcurrent/QtConcurrent>

#include <data/analysis/ftworker.h>
#include <gui/widget/peakfindwidget.h>
#include <data/storage/fidsinglestorage.h>
#include <data/storage/fidpeakupstorage.h>
#include <data/storage/fidmultistorage.h>

FtmwViewWidget::FtmwViewWidget(bool main, QWidget *parent, QString path) :
    QWidget(parent), SettingsStorage(BC::Key::FtmwView::key),
    ui(new Ui::FtmwViewWidget), d_currentExptNum(-1), d_currentSegment(-1), d_path(path)
{
    ui->setupUi(main,this);

    d_currentProcessingSettings = ui->processingToolBar->getSettings();
    connect(ui->processingToolBar,&FtmwProcessingToolBar::resetSignal,this,&FtmwViewWidget::resetProcessingSettings);
    connect(ui->processingToolBar,&FtmwProcessingToolBar::saveSignal,this,&FtmwViewWidget::saveProcessingSettings);
    p_worker = new FtWorker(this);
    connect(p_worker,&FtWorker::ftDone,this,&FtmwViewWidget::ftDone,Qt::QueuedConnection);
    connect(p_worker,&FtWorker::fidDone,this,&FtmwViewWidget::fidProcessed,Qt::QueuedConnection);
    connect(p_worker,&FtWorker::ftDiffDone,this,&FtmwViewWidget::ftDiffDone,Qt::QueuedConnection);
    connect(p_worker,&FtWorker::sidebandDone,this,&FtmwViewWidget::sidebandProcessingComplete);

    d_workerIds << d_liveId << d_mainId << d_plot1Id << d_plot2Id;

    for(int i=0; i<d_workerIds.size(); i++)
    {
        int id = d_workerIds.at(i);

        auto fw = new QFutureWatcher<void>(this);
        connect(fw,&QFutureWatcher<void>::finished,[this,id]{
            ftProcessingComplete(id);
        });
        d_workersStatus.emplace(id,WorkerStatus{ fw, false, false });

        if(id != d_mainId)
        {
            auto fw2 = new QFutureWatcher<FidList>(this);
            connect(fw2,&QFutureWatcher<FidList>::finished,[this,id](){
                fidLoadComplete(id);
            });
            if(id == d_liveId)
                d_plotStatus.emplace(id,PlotStatus { fw2, ui->liveFidPlot, ui->liveFtPlot, FidList(), Ft() });
            else if(id == d_plot1Id)
                d_plotStatus.emplace(id,PlotStatus { fw2, ui->fidPlot1, ui->ftPlot1, FidList(), Ft() });
            else if(id == d_plot2Id)
                d_plotStatus.emplace(id,PlotStatus { fw2, ui->fidPlot2, ui->ftPlot2, FidList(), Ft() });
            //don't need to add one of these for the main plot; it's special
        }

    }

    d_sbStatus.sbLoadWatcher = new QFutureWatcher<FidList>(this);
    connect(d_sbStatus.sbLoadWatcher,&QFutureWatcher<FidList>::finished,this,&FtmwViewWidget::sidebandLoadComplete);

    for(auto &[key,ps] : d_plotStatus)
    {
        (void)key;
        ps.fidPlot->blockSignals(true);
        ps.fidPlot->setFtStart(d_currentProcessingSettings.startUs);
        ps.fidPlot->setFtEnd(d_currentProcessingSettings.endUs);
        ps.fidPlot->blockSignals(false);
    }

    connect(ui->processingToolBar,&FtmwProcessingToolBar::settingsUpdated,this,&FtmwViewWidget::updateProcessingSettings);
    connect(ui->processingAct,&QAction::triggered,ui->processingToolBar,&FtmwProcessingToolBar::setVisible);

    connect(ui->plotAction,&QAction::triggered,ui->plotToolBar,&FtmwPlotToolBar::setVisible);
    connect(ui->plotToolBar,&FtmwPlotToolBar::mainPlotSettingChanged,this,&FtmwViewWidget::updateMainPlot);
    connect(ui->plotToolBar,&FtmwPlotToolBar::plotSettingChanged,this,&FtmwViewWidget::updatePlotSetting);


    ui->refreshBox->setValue(get(BC::Key::FtmwView::refresh,500));
    registerGetter(BC::Key::FtmwView::refresh,ui->refreshBox,&SpinBoxWidgetAction::value);
    ui->refreshBox->setEnabled(false);

    ui->processingToolBar->setEnabled(false);
    ui->plotToolBar->setEnabled(false);


    connect(ui->peakFindAction,&QAction::triggered,this,&FtmwViewWidget::launchPeakFinder);

    connect(ui->averagesSpinbox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwViewWidget::changeRollingAverageShots,Qt::UniqueConnection);
    connect(ui->resetAveragesButton,&QPushButton::clicked,this,&FtmwViewWidget::resetRollingAverage,Qt::UniqueConnection);

}

FtmwViewWidget::~FtmwViewWidget()
{
    clearGetters();

    if(p_pfw != nullptr)
        p_pfw->close();

    d_sbStatus.sbLoadWatcher->waitForFinished();

    for(auto &[key,ps] : d_plotStatus)
    {
        Q_UNUSED(key)
        ps.p_watcher->waitForFinished();
    }

    for(auto &[key,ws] : d_workersStatus)
    {
        Q_UNUSED(key)
        ws.p_watcher->waitForFinished();
    }


    delete ui;
}

void FtmwViewWidget::prepareForExperiment(const Experiment &e)
{
    if(p_pfw != nullptr)
    {
        p_pfw->close();
        p_pfw = nullptr;
    }

    if(!ui->exptLabel->isVisible())
        ui->exptLabel->setVisible(true);

    ui->refreshBox->setEnabled(false);

    ui->liveFidPlot->prepareForExperiment(e);
    ui->liveFidPlot->setVisible(true);

    ui->fidPlot1->prepareForExperiment(e);
    ui->fidPlot2->prepareForExperiment(e);

//    ui->peakFindWidget->prepareForExperiment(e);

    ui->liveFtPlot->prepareForExperiment(e);
    ui->processingToolBar->prepareForExperient(e);
    ui->plotToolBar->prepareForExperiment(e);
    ui->ftPlot1->prepareForExperiment(e);
    ui->ftPlot2->prepareForExperiment(e);
    ui->mainFtPlot->prepareForExperiment(e);

    d_currentSegment = 0;
    for(auto &[key,ps] : d_plotStatus)
    {
        Q_UNUSED(key)
        ps.fidList.clear();
        ps.ft = Ft();
        ps.frame = 0;
        ps.segment = 0;
        ps.backup = 0;
        ps.loadWhenDone = false;
    }

    if(e.ftmwEnabled())
    {        
        ui->refreshBox->setEnabled(true);
        connect(ui->refreshBox,&SpinBoxWidgetAction::valueChanged,this,&FtmwViewWidget::setLiveUpdateInterval);
        ps_fidStorage = e.ftmwConfig()->storage();
        if(e.ftmwConfig()->d_type == FtmwConfig::Peak_Up)
            ui->exptLabel->setText(QString("Peak Up Mode"));
        else
            ui->exptLabel->setText(QString("Experiment %1").arg(e.d_number));

        d_currentExptNum = e.d_number;

        ui->verticalLayout->setStretch(0,1);
        ui->liveFidPlot->show();
        ui->liveFtPlot->show();

        ui->averagesSpinbox->blockSignals(true);
        ui->averagesSpinbox->setValue(e.ftmwConfig()->d_type == FtmwConfig::Peak_Up ? e.ftmwConfig()->d_objective : 0);
        ui->averagesSpinbox->blockSignals(false);

        ui->resetAveragesButton->setEnabled(e.ftmwConfig()->d_type == FtmwConfig::Peak_Up);
        ui->averagesSpinbox->setEnabled(e.ftmwConfig()->d_type == FtmwConfig::Peak_Up);

        d_liveTimerId = startTimer(ui->refreshBox->value());
    }
    else
    {
        ps_fidStorage.reset();
        ui->exptLabel->setText(QString("Experiment %1").arg(e.d_number));
        ui->resetAveragesButton->setEnabled(false);
        ui->averagesSpinbox->setEnabled(false);
    }

    ui->peakFindAction->setEnabled(false);

}

void FtmwViewWidget::setLiveUpdateInterval(int intervalms)
{
    if(d_liveTimerId >= 0)
        killTimer(d_liveTimerId);

    d_liveTimerId = startTimer(intervalms);
}

void FtmwViewWidget::updateLiveFidList()
{
    auto fl = ps_fidStorage->getCurrentFidList();
    if(fl.isEmpty())
        return;

    d_currentSegment = ps_fidStorage->getCurrentIndex();

    for(auto &[key,ps] : d_plotStatus)
    {
        if(key != d_liveId)
        {
            if(d_currentSegment == ps.segment && ps.frame < fl.size())
            {
                if(!ui->plotToolBar->viewingBackup(key))
                {
                    ps.fidList = fl;
                    process(key,fl,ps.frame);
                }
            }
        }
        else
        {
            //always average all frames for live plot
            ps.fidList = fl;
            process(key,fl,-1);
        }

    }
}

void FtmwViewWidget::updateProcessingSettings(FtWorker::FidProcessingSettings s)
{
    //skip main plot because it will be updated when menu is closed
    d_currentProcessingSettings = s;
    QList<int> ignore;
//    switch(ui->plotToolBar->mainPlotMode())
//    {
//    case FtmwPlotToolBar::Upper_SideBand:
//    case FtmwPlotToolBar::Lower_SideBand:
//    case FtmwPlotToolBar::Both_SideBands:
//        ignore << d_mainId;
//    default:
//        break;
//    }

    if(!ui->liveFidPlot->isHidden())
    {
        ui->liveFidPlot->setFtStart(s.startUs);
        ui->liveFidPlot->setFtEnd(s.endUs);
    }
    else
        ignore << d_liveId;

    ui->fidPlot1->setFtStart(s.startUs);
    ui->fidPlot1->setFtEnd(s.endUs);
    ui->fidPlot2->setFtStart(s.startUs);
    ui->fidPlot2->setFtEnd(s.endUs);

    reprocess(ignore);
}

void FtmwViewWidget::resetProcessingSettings()
{
    if(ps_fidStorage)
    {
        if(ps_fidStorage->readProcessingSettings(d_currentProcessingSettings))
            ui->processingToolBar->setAll(d_currentProcessingSettings);
    }
}

void FtmwViewWidget::saveProcessingSettings()
{
    if(ps_fidStorage)
        ps_fidStorage->writeProcessingSettings(ui->processingToolBar->getSettings());
}

void FtmwViewWidget::updatePlotSetting(int id)
{
    auto it = d_plotStatus.find(id);
    if(it != d_plotStatus.end())
    {
        //segment and frame are 1-indexed on the UI
        it->second.segment = ui->plotToolBar->segment(id)-1;
        it->second.frame = ui->plotToolBar->frame(id)-1;
        it->second.backup = ui->plotToolBar->backup(id);
        updateFid(id);
    }
}

void FtmwViewWidget::fidLoadComplete(int id)
{
    auto &ps = d_plotStatus[id];
    if(ps.loadWhenDone)
    {
        ps.loadWhenDone = false;
        updateFid(id);
    }
    else
    {
        ps.fidList = ps.p_watcher->result();
        process(id, ps.fidList, ps.frame);
    }
}

void FtmwViewWidget::ftProcessingComplete(int id)
{
    auto &ws = d_workersStatus[id];
    ws.busy = false;
    if(ws.reprocessWhenDone) //this is set to true when there is another FID to process
    {
        if(id == d_mainId)
        {
            switch(ui->plotToolBar->mainPlotMode())
            {
            case FtmwPlotToolBar::Lower_SideBand:
            case FtmwPlotToolBar::Upper_SideBand:
            case FtmwPlotToolBar::Both_SideBands:
                if(d_sbStatus.cancel)
                    updateMainPlot();
                else
                {
                    if(!d_sbStatus.nextFidList.isEmpty())
                        processNextSidebandFid();
                    if(!d_sbStatus.sbLoadWatcher->isRunning())
                        loadNextSidebandFid();
                }
                break;
            default:
                updateMainPlot();
                break;
            }
        }
        else
            process(id,d_plotStatus[id].fidList,d_plotStatus[id].frame);
    }
}

void FtmwViewWidget::fidProcessed(const QVector<double> fidData, double spacing, double min, double max, quint64 shots, int workerId)
{
    auto it = d_plotStatus.find(workerId);
    if(it != d_plotStatus.end())
    {
        auto &ps = it->second;
        if(!ps.fidPlot->isHidden())
            ps.fidPlot->receiveProcessedFid(fidData,spacing,min,max,shots);
    }
}

void FtmwViewWidget::ftDone(const Ft ft, int workerId)
{
    auto it = d_plotStatus.find(workerId);
    if(it != d_plotStatus.end())
    {
        auto &ps = it->second;
        if(!ps.ftPlot->isHidden())
        {
            ps.ft = ft;
            ps.ftPlot->configureUnits(d_currentProcessingSettings.units);
            ps.ftPlot->newFt(ft);
        }

        ps.fidPlot->setCursor(Qt::CrossCursor);
        ps.ftPlot->setCursor(Qt::CrossCursor);

        switch(ui->plotToolBar->mainPlotMode()) {
        case FtmwPlotToolBar::Live:
        case FtmwPlotToolBar::FT1:
        case FtmwPlotToolBar::FT2:
        case FtmwPlotToolBar::FT1_minus_FT2:
        case FtmwPlotToolBar::FT2_minus_FT1:
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
        ui->peakFindAction->setEnabled(!ft.isEmpty());
        ui->mainFtPlot->canvas()->setCursor(QCursor(Qt::CrossCursor));
        if(p_pfw != nullptr)
            p_pfw->newFt(ft);
    }
}

void FtmwViewWidget::ftDiffDone(const Ft ft)
{
    ui->mainFtPlot->newFt(ft);
    ui->mainFtPlot->canvas()->setCursor(QCursor(Qt::CrossCursor));
}

void FtmwViewWidget::updateMainPlot()
{
    ui->mainFtPlot->configureUnits(d_currentProcessingSettings.units);
    ui->mainFtPlot->setMessageText("");
    ui->mainFtPlot->canvas()->setCursor(QCursor(Qt::CrossCursor));

    cancelSidebandProcessing();

    switch(ui->plotToolBar->mainPlotMode()) {
    case FtmwPlotToolBar::Live:
        ui->mainFtPlot->newFt(d_plotStatus[d_liveId].ft);
        if(p_pfw != nullptr)
            p_pfw->newFt(d_plotStatus[d_liveId].ft);
        break;
    case FtmwPlotToolBar::FT1:
        ui->mainFtPlot->newFt(d_plotStatus[d_plot1Id].ft);
        if(p_pfw != nullptr)
            p_pfw->newFt(d_plotStatus[d_plot1Id].ft);
        break;
    case FtmwPlotToolBar::FT2:
        ui->mainFtPlot->newFt(d_plotStatus[d_plot2Id].ft);
        if(p_pfw != nullptr)
            p_pfw->newFt(d_plotStatus[d_plot2Id].ft);
        break;
    case FtmwPlotToolBar::FT1_minus_FT2:
        processDiff(d_plotStatus[d_plot1Id].fidList,d_plotStatus[d_plot2Id].fidList,
                    d_plotStatus[d_plot1Id].frame,d_plotStatus[d_plot2Id].frame);
        break;
    case FtmwPlotToolBar::FT2_minus_FT1:
        processDiff(d_plotStatus[d_plot2Id].fidList,d_plotStatus[d_plot1Id].fidList,
                    d_plotStatus[d_plot2Id].frame,d_plotStatus[d_plot1Id].frame);
        break;
    case FtmwPlotToolBar::Upper_SideBand:
    case FtmwPlotToolBar::Lower_SideBand:
    case FtmwPlotToolBar::Both_SideBands:
        processSidebands();
        break;
    }

    ui->peakFindAction->setEnabled(!ui->mainFtPlot->currentFt().isEmpty());
}

void FtmwViewWidget::reprocess(const QList<int> ignore)
{
    for(auto &[key,ws] : d_workersStatus)
    {
        Q_UNUSED(ws)
        if(!ignore.contains(key))
        {
            if(key == d_mainId)
                updateMainPlot();
            else if(key == d_liveId)
                process(key,d_plotStatus[key].fidList,-1);
            else
                process(key,d_plotStatus[key].fidList,d_plotStatus[key].frame);
        }
    }
}

void FtmwViewWidget::process(int id, const FidList fl, int frame)
{
//    if(f.isEmpty())
//        return;
    auto &ws = d_workersStatus[id];
    if(ws.busy)
        ws.reprocessWhenDone = true;
    else
    {
        d_plotStatus[id].fidPlot->setCursor(Qt::BusyCursor);
        d_plotStatus[id].ftPlot->setCursor(Qt::BusyCursor);
        ws.busy = true;
        ws.reprocessWhenDone = false;
        ws.p_watcher->setFuture(QtConcurrent::run([fl,frame,id,this](){
            p_worker->doFT(fl,d_currentProcessingSettings,frame,id);
        }));
    }
}

void FtmwViewWidget::processDiff(const FidList fl1, const FidList fl2, int frame1, int frame2)
{
    if(fl1.isEmpty() || fl2.isEmpty())
        return;

    auto &ws = d_workersStatus[d_mainId];
    if(ws.busy)
        ws.reprocessWhenDone = true;
    else
    {
        ui->mainFtPlot->canvas()->setCursor(QCursor(Qt::BusyCursor));
        ws.busy = true;
        ws.reprocessWhenDone = false;
        ws.p_watcher->setFuture(QtConcurrent::run([fl1,fl2,frame1,frame2,this](){
            p_worker->doFtDiff(fl1,fl2,frame1,frame2,d_currentProcessingSettings);
        }));

    }
}

void FtmwViewWidget::sidebandLoadComplete()
{
    d_sbStatus.nextFidList = FidList();

    if(d_sbStatus.cancel)
    {
        updateMainPlot();
        return;
    }
    auto fl = d_sbStatus.sbLoadWatcher->result();

    //queue FID
    d_sbStatus.nextFidList = fl;
    if(!d_workersStatus[d_mainId].busy)
    {
        processNextSidebandFid();
        loadNextSidebandFid();
    }

}

void FtmwViewWidget::processSidebands()
{  
    auto &ws = d_workersStatus[d_mainId];
    if(ws.busy)
        ws.reprocessWhenDone = true;
    else
    {
        //need to reset sideband parameters
        //then need to set up load/process chain
        auto storage = dynamic_cast<FidMultiStorage*>(ps_fidStorage.get());
        if(!storage)
            return;

        if(d_sbStatus.sbLoadWatcher->isRunning())
        {
            cancelSidebandProcessing();
            return;
        }

        d_sbStatus.cancel = false;
        d_sbStatus.complete = false;
        auto &sbd = d_sbStatus.sbData;
        sbd = FtWorker::SidebandProcessingData();
        sbd.frame = ui->plotToolBar->sbFrame()-1;
        sbd.minOffset = ui->plotToolBar->sbMinFreq();
        sbd.maxOffset = ui->plotToolBar->sbMaxFreq();
        sbd.dcMethod = ui->plotToolBar->dcMethod();
        sbd.totalFids = storage->numSegments();
        sbd.loRange = storage->getLORange();
        switch (ui->plotToolBar->mainPlotMode()) {
        case FtmwPlotToolBar::Lower_SideBand:
            sbd.doubleSideband = false;
            sbd.sideband = RfConfig::LowerSideband;
            break;
        case FtmwPlotToolBar::Upper_SideBand:
            sbd.doubleSideband = false;
            sbd.sideband = RfConfig::UpperSideband;
            break;
        case FtmwPlotToolBar::Both_SideBands:
            sbd.doubleSideband  = true;
            break;
        default:
            break;
        };
        d_sbStatus.sbData = sbd;

        ui->mainFtPlot->canvas()->setCursor(QCursor(Qt::BusyCursor));
        ui->mainFtPlot->setMessageText(QString("Processing..."));
        ui->mainFtPlot->newFt(Ft());
        loadNextSidebandFid();
    }
}

void FtmwViewWidget::loadNextSidebandFid()
{
    if(d_sbStatus.cancel && !d_sbStatus.complete)
    {
        updateMainPlot();
        return;
    }

    if(d_sbStatus.sbData.currentIndex >= d_sbStatus.sbData.totalFids)
        return;

    d_sbStatus.sbLoadWatcher->setFuture(QtConcurrent::run([this](){ return ps_fidStorage->loadFidList(d_sbStatus.sbData.currentIndex); }));

}

void FtmwViewWidget::processNextSidebandFid()
{
    auto fl = d_sbStatus.nextFidList;
    d_sbStatus.nextFidList = FidList();

    if(d_sbStatus.cancel || d_sbStatus.sbData.currentIndex >= d_sbStatus.sbData.totalFids)
        return;

    auto &ws = d_workersStatus[d_mainId];
    ws.busy = true;
    ws.reprocessWhenDone = true;
    d_sbStatus.sbData.fl= fl;
    auto sbd = d_sbStatus.sbData;
    ws.p_watcher->setFuture(QtConcurrent::run([this,sbd]{
        p_worker->processSideband2(sbd,d_currentProcessingSettings);
    }));
    d_sbStatus.sbData.currentIndex++;
    ui->mainFtPlot->setMessageText(QString("Processing %1/%2")
                                   .arg(d_sbStatus.sbData.currentIndex)
                                   .arg(d_sbStatus.sbData.totalFids));
    ui->mainFtPlot->replot();
}

void FtmwViewWidget::sidebandProcessingComplete(const Ft ft)
{
    d_sbStatus.complete = true;

    if(d_sbStatus.cancel)
        updateMainPlot();
    else
    {
        d_sbStatus.nextFidList = FidList();

        ui->mainFtPlot->canvas()->setCursor(QCursor(Qt::CrossCursor));
        ui->mainFtPlot->setMessageText("");
        ui->mainFtPlot->newFt(ft);
    }
}

void FtmwViewWidget::cancelSidebandProcessing()
{
    d_sbStatus.cancel = true;
    d_sbStatus.nextFidList = FidList();
}

void FtmwViewWidget::updateBackups()
{
    if(d_currentExptNum < 1)
        return;

    ui->plotToolBar->newBackup(ps_fidStorage->numBackups());
}

void FtmwViewWidget::experimentComplete()
{
    disconnect(ui->refreshBox,&SpinBoxWidgetAction::valueChanged,this,&FtmwViewWidget::setLiveUpdateInterval);
    ui->refreshBox->setEnabled(false);
    if(d_liveTimerId >= 0)
        killTimer(d_liveTimerId);
    d_liveTimerId = -1;   

    if(ps_fidStorage)
    {
        d_currentSegment = -1;

        ui->verticalLayout->setStretch(0,0);
        ui->liveFidPlot->hide();
        ui->liveFtPlot->hide();

        ui->plotToolBar->experimentComplete();

        updateFid(d_plot1Id);
        updateFid(d_plot2Id);
        updateMainPlot();
    }

}

void FtmwViewWidget::changeRollingAverageShots(int shots)
{
    if(shots < 1)
        return;

    auto p = dynamic_cast<FidPeakUpStorage*>(ps_fidStorage.get());
    if(p != nullptr)
        p->setTargetShots(static_cast<quint64>(shots));
}

void FtmwViewWidget::resetRollingAverage()
{
    auto p = dynamic_cast<FidPeakUpStorage*>(ps_fidStorage.get());
    if(p != nullptr)
        p->reset();
}

void FtmwViewWidget::launchPeakFinder()
{
    if(p_pfw != nullptr)
    {
        p_pfw->activateWindow();
        p_pfw->raise();
        return;
    }

    p_pfw = new PeakFindWidget(ui->mainFtPlot->currentFt(),d_currentExptNum);
    if(d_currentExptNum > 0)
        p_pfw->setWindowTitle(QString("Peak List: Experiment %1").arg(d_currentExptNum));
    else
        p_pfw->setWindowTitle(QString("Peak List"));

    p_pfw->setAttribute(Qt::WA_DeleteOnClose);

    connect(p_worker,&FtWorker::ftDone,[this](const Ft ft, int id){
        if(id == d_mainId)
            p_pfw->newFt(ft);
    });
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
    if(id == d_mainId)
        return;

    auto &ps = d_plotStatus[id];
    int seg = ps.segment;
    int backup = ps.backup;

    if(seg == d_currentSegment && id == d_liveId)
    {
        //For live plots, always average all frames
        auto fl = ps_fidStorage->getCurrentFidList();
        ps.fidList = fl;
        process(id, ps.fidList, -1);
    }
    else
    {
        // only acquisition modes with 1 segment have backups right now... might change this later!
        if(backup > 0)
            seg = backup;

        if(ps.p_watcher->isRunning())
            ps.loadWhenDone = true;
        else
            ps.p_watcher->setFuture(QtConcurrent::run([this,seg](){ return ps_fidStorage->loadFidList(seg); }));
    }
}



void FtmwViewWidget::timerEvent(QTimerEvent *event)
{
    if(event->timerId() == d_liveTimerId)
    {
        updateLiveFidList();
        event->accept();
    }
}
