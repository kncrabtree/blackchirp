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

FtmwViewWidget::FtmwViewWidget(QWidget *parent, QString path) :
    QWidget(parent),
    ui(new Ui::FtmwViewWidget), d_currentExptNum(-1), d_currentSegment(-1), d_path(path)
{
    ui->setupUi(this);

    p_pfw = nullptr;

    d_currentProcessingSettings = ui->processingToolBar->getSettings();

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
            d_plotStatus.emplace(id,PlotStatus { ui->liveFidPlot, ui->liveFtPlot, Fid(), Ft() });
        else if(id == d_plot1Id)
            d_plotStatus.emplace(id,PlotStatus { ui->fidPlot1, ui->ftPlot1, Fid(), Ft() });
        else if(id == d_plot2Id)
            d_plotStatus.emplace(id,PlotStatus { ui->fidPlot2, ui->ftPlot2, Fid(), Ft() });
        //don't need to add one of these for the main plot; it's special

    }

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

    ui->processingToolBar->setEnabled(false);
    ui->plotToolBar->setEnabled(false);


    connect(ui->peakFindAction,&QAction::triggered,this,&FtmwViewWidget::launchPeakFinder);

    connect(ui->averagesSpinbox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&FtmwViewWidget::changeRollingAverageShots,Qt::UniqueConnection);
    connect(ui->resetAveragesButton,&QPushButton::clicked,this,&FtmwViewWidget::resetRollingAverage,Qt::UniqueConnection);

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

void FtmwViewWidget::prepareForExperiment(const Experiment &e)
{
    if(p_pfw != nullptr)
    {
        p_pfw->close();
        p_pfw = nullptr;
    }

    if(!ui->exptLabel->isVisible())
        ui->exptLabel->setVisible(true);

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
        ps.fid = Fid();
        ps.ft = Ft();
        ps.frame = 0;
        ps.segment = 0;
        ps.backup = 0;
        ps.loadWhenDone = false;
        ps.pu_watcher.reset();
        ps.pu_watcher = std::make_unique<QFutureWatcher<FidList>>();
        int id = key;
        connect(ps.pu_watcher.get(),&QFutureWatcher<FidList>::finished,[this,id](){ fidLoadComplete(id); });
    }

    if(e.ftmwEnabled())
    {        
        ps_fidStorage = e.ftmwConfig()->storage();
        if(e.ftmwConfig()->d_type == FtmwConfig::Peak_Up)
            ui->exptLabel->setText(QString("Peak Up Mode"));
        else
            ui->exptLabel->setText(QString("Experiment %1").arg(e.d_number));

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

        ui->averagesSpinbox->blockSignals(true);
        ui->averagesSpinbox->setValue(e.ftmwConfig()->d_type == FtmwConfig::Peak_Up ? e.ftmwConfig()->d_objective : 0);
        ui->averagesSpinbox->blockSignals(false);

        ui->resetAveragesButton->setEnabled(e.ftmwConfig()->d_type == FtmwConfig::Peak_Up);
        ui->averagesSpinbox->setEnabled(e.ftmwConfig()->d_type == FtmwConfig::Peak_Up);

        d_liveTimerId = startTimer(500);
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

void FtmwViewWidget::updateLiveFidList()
{
    auto fl = ps_fidStorage->getCurrentFidList();
    if(fl.isEmpty())
        return;

    d_currentSegment = ps_fidStorage->getCurrentIndex();

    for(auto &[key,ps] : d_plotStatus)
    {
        if(d_workersStatus.value(key).thread->isRunning())
        {
            if(key != d_liveId)
            {
                if(d_currentSegment == ps.segment && ps.frame < fl.size())
                {
                    if(!ui->plotToolBar->viewingBackup(key))
                    {
                        auto f = fl.at(ps.frame);
                        ps.fid = f;
                        process(key,f);
                    }
                }
            }
            else
            {
                auto f = fl.constFirst();
                ps.fid = f;
                process(key,f);
            }

        }
    }
}

void FtmwViewWidget::updateProcessingSettings(FtWorker::FidProcessingSettings s)
{
    //skip main plot because it will be updated when menu is closed
    d_currentProcessingSettings = s;
    QList<int> ignore;
    switch(ui->plotToolBar->mainPlotMode())
    {
    case FtmwPlotToolBar::Upper_SideBand:
    case FtmwPlotToolBar::Lower_SideBand:
    case FtmwPlotToolBar::Both_SideBands:
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

void FtmwViewWidget::updatePlotSetting(int id)
{
    auto it = d_plotStatus.find(id);
    if(it != d_plotStatus.end())
    {
        it->second.segment = ui->plotToolBar->segment(id);
        it->second.frame = ui->plotToolBar->frame(id);
        it->second.backup = ui->plotToolBar->backup(id);
        updateFid(id);
    }
}

void FtmwViewWidget::fidLoadComplete(int id)
{
#pragma message("Sideband processing here")
    auto &ps = d_plotStatus[id];
    if(ps.loadWhenDone)
    {
        ps.loadWhenDone = false;
        updateFid(id);
    }
    else
    {
        auto list = ps.pu_watcher->result();
        ps.fid = list.at(ps.frame);
        process(id, ps.fid);
    }
}

void FtmwViewWidget::fidProcessed(const QVector<QPointF> fidData, int workerId)
{
    auto it = d_plotStatus.find(workerId);
    if(it != d_plotStatus.end())
    {
        auto &ps = it->second;
        if(!ps.fidPlot->isHidden())
            ps.fidPlot->receiveProcessedFid(fidData);
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
            if(ui->plotToolBar->viewingBackup(workerId) && ui->plotToolBar->mainPlotFollow() == workerId)
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
            process(workerId,d_plotStatus[workerId].fid);
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
        processDiff(d_plotStatus[d_plot1Id].fid,d_plotStatus[d_plot2Id].fid);
        break;
    case FtmwPlotToolBar::FT2_minus_FT1:
        processDiff(d_plotStatus[d_plot2Id].fid,d_plotStatus[d_plot1Id].fid);
        break;
    case FtmwPlotToolBar::Upper_SideBand:
        processSideband(RfConfig::UpperSideband);
        break;
    case FtmwPlotToolBar::Lower_SideBand:
        processSideband(RfConfig::LowerSideband);
        break;
    case FtmwPlotToolBar::Both_SideBands:
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
                process(it.key(),d_plotStatus[it.key()].fid);
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
            d_plotStatus[id].fidPlot->setNumShots(f.shots());
            d_plotStatus[id].fidPlot->setCursor(Qt::BusyCursor);
            d_plotStatus[id].ftPlot->setCursor(Qt::BusyCursor);
            d_workersStatus[id].busy = true;
            d_workersStatus[id].reprocessWhenDone = false;
            QMetaObject::invokeMethod(ws.worker,[ws,f,this](){
                ws.worker->doFT(f,d_currentProcessingSettings);
            });
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
        QMetaObject::invokeMethod(ws.worker,[ws,f1,f2,this](){
            ws.worker->doFtDiff(f1,f2,d_currentProcessingSettings);
        });

    }
}

void FtmwViewWidget::processSideband(RfConfig::Sideband sb)
{
    auto ws = d_workersStatus.value(d_mainId);
    if(ws.busy)
        d_workersStatus[d_mainId].reprocessWhenDone = true;
    else
    {
        FidList fl;

#pragma message("Here")
//        int id = d_plot1Id;
//        if(ui->mainPlotFollowSpinBox->value() == 2)
//            id = d_plot2Id;


#pragma message("Rework sideband processing so that it's not monolithic")
//        auto n = p_fidStorage->d_numRecords;
//        for(int i=0; i<n; i++)
//            fl << p_fidStorage->loadFidList(i).at(d_plotStatus.value(id).frame);

        if(!fl.isEmpty())
        {
            ui->mainFtPlot->canvas()->setCursor(QCursor(Qt::BusyCursor));
            d_workersStatus[d_mainId].busy = true;
            d_workersStatus[d_mainId].reprocessWhenDone = false;
            double minF = ui->minFtSegBox->value();
            double maxF = ui->maxFtSegBox->value();

            QMetaObject::invokeMethod(ws.worker,[ws,fl,this,sb,minF,maxF](){
                ws.worker->processSideband(fl,d_currentProcessingSettings,sb,minF,maxF);
            });
        }
    }
}

void FtmwViewWidget::processBothSidebands()
{
#pragma message("Rework sideband processing so that it's not monolithic")
    auto ws = d_workersStatus.value(d_mainId);
    if(ws.busy)
        d_workersStatus[d_mainId].reprocessWhenDone = true;
    else
    {
        FidList fl;

#pragma message("Here")
//        int id = d_plot1Id;
//        if(ui->mainPlotFollowSpinBox->value() == 2)
//            id = d_plot2Id;


//        auto n = p_fidStorage->d_numRecords;
//        for(int i=0; i<n; i++)
//            fl << p_fidStorage->loadFidList(i).at(d_plotStatus.value(id).frame);

        if(!fl.isEmpty())
        {
            ui->mainFtPlot->canvas()->setCursor(QCursor(Qt::BusyCursor));
            d_workersStatus[d_mainId].busy = true;
            d_workersStatus[d_mainId].reprocessWhenDone = false;
            double minF = ui->minFtSegBox->value();
            double maxF = ui->maxFtSegBox->value();

            QMetaObject::invokeMethod(ws.worker,[ws,fl,this,minF,maxF](){
                ws.worker->processBothSidebands(fl,d_currentProcessingSettings,minF,maxF);
            });
        }
    }
}

void FtmwViewWidget::updateBackups()
{
    if(d_currentExptNum < 1)
        return;

    int n = ps_fidStorage->numBackups();
#pragma message("Here")
//    ui->plot1ConfigWidget->newBackup(n);
//    ui->plot2ConfigWidget->newBackup(n);
}

void FtmwViewWidget::experimentComplete()
{
    killTimer(d_liveTimerId);
    if(ps_fidStorage)
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

    ///TODO update UI?
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
    if(id == d_mainId)
        return;

    auto &ps = d_plotStatus[id];
    int seg = ps.segment;
    int frame = ps.frame;
    int backup = ps.backup;

    if(seg == d_currentSegment && id == d_liveId)
    {
        auto fl = ps_fidStorage->getCurrentFidList();
        if(frame >= 0 && frame < fl.size())
            ps.fid = fl.at(frame);

        process(id, ps.fid);
    }
    else
    {
        // only acquisition modes with 1 segment have backups right now... might change this later!
        if(backup > 0)
            seg = backup;

        if(ps.pu_watcher->isRunning())
            ps.loadWhenDone = true;
        else
            ps.pu_watcher->setFuture(QtConcurrent::run([this,seg](){ return ps_fidStorage->loadFidList(seg); }));
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
