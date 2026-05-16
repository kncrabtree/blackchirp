#include <gui/widget/ftmwviewwidget.h>

#include <QAction>
#include <QByteArray>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDockWidget>
#include <QHBoxLayout>
#include <QLabel>
#include <QMainWindow>
#include <QMessageBox>
#include <QPushButton>
#include <QSplitter>
#include <QThread>
#include <QToolBar>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrent>

#include <data/analysis/ftworker.h>
#include <data/storage/fidsinglestorage.h>
#include <data/storage/fidpeakupstorage.h>
#include <data/storage/fidmultistorage.h>
#include <gui/overlay/overlaymanagerwidget.h>
#include <gui/plot/fidplot.h>
#include <gui/plot/ftplot.h>
#include <gui/plot/mainftplot.h>

#include <qwt6/qwt_interval.h>
#include <gui/style/themecolors.h>
#include <gui/widget/ftmwacquisitionpanel.h>
#include <gui/widget/ftmwplotpanel.h>
#include <gui/widget/ftmwprocessingpanel.h>
#include <gui/widget/peakfindwidget.h>


FtmwViewWidget::FtmwViewWidget(bool main, QWidget *parent, QString path, bool overlaysEnabled) :
    QWidget(parent), SettingsStorage(BC::Key::FtmwView::key),
    d_main(main), d_overlaysEnabled(overlaysEnabled),
    d_currentExptNum(-1), d_currentSegment(-1), d_path(path)
{
    setupInnerUi();

    d_currentProcessingSettings = p_processingPanel->getSettings();
    connect(p_processingPanel,&FtmwProcessingPanel::resetSignal,this,&FtmwViewWidget::resetProcessingSettings);
    connect(p_processingPanel,&FtmwProcessingPanel::saveSignal,this,&FtmwViewWidget::saveProcessingSettings);
    connect(p_processingPanel,&FtmwProcessingPanel::settingsUpdated,this,&FtmwViewWidget::updateProcessingSettings);

    p_worker = new FtWorker(this);
    connect(p_worker,&FtWorker::ftDone,this,&FtmwViewWidget::ftDone,Qt::QueuedConnection);
    connect(p_worker,&FtWorker::fidDone,this,&FtmwViewWidget::fidProcessed,Qt::QueuedConnection);
    connect(p_worker,&FtWorker::ftDiffDone,this,&FtmwViewWidget::ftDiffDone,Qt::QueuedConnection);
    connect(p_worker,&FtWorker::sidebandDone,this,&FtmwViewWidget::sidebandProcessingComplete);

    if(!d_main)
        p_worker->setIdleCleanupEnabled(true);

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
                d_plotStatus.emplace(id,PlotStatus { fw2, p_liveFidPlot, p_liveFtPlot, FidList(), Ft() });
            else if(id == d_plot1Id)
                d_plotStatus.emplace(id,PlotStatus { fw2, p_fidPlot1, p_ftPlot1, FidList(), Ft() });
            else if(id == d_plot2Id)
                d_plotStatus.emplace(id,PlotStatus { fw2, p_fidPlot2, p_ftPlot2, FidList(), Ft() });
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

    connect(p_plotPanel,&FtmwPlotPanel::mainPlotSettingChanged,this,&FtmwViewWidget::updateMainPlot);
    connect(p_plotPanel,&FtmwPlotPanel::plotSettingChanged,this,&FtmwViewWidget::updatePlotSetting);

    if(p_acquisitionPanel)
    {
        p_acquisitionPanel->setRefreshInterval(get(BC::Key::FtmwView::refresh,500));
        registerGetter(BC::Key::FtmwView::refresh,p_acquisitionPanel,&FtmwAcquisitionPanel::refreshInterval);
        p_acquisitionPanel->setRefreshEnabled(false);

        connect(p_acquisitionPanel,&FtmwAcquisitionPanel::averagesChanged,this,&FtmwViewWidget::changeRollingAverageShots,Qt::UniqueConnection);
        connect(p_acquisitionPanel,&FtmwAcquisitionPanel::resetAveragesClicked,this,&FtmwViewWidget::resetRollingAverage,Qt::UniqueConnection);
        connect(p_acquisitionPanel,&FtmwAcquisitionPanel::manualBackupClicked,this,&FtmwViewWidget::manualBackupRequested);
    }

    p_processingPanel->setEnabled(false);
    p_plotPanel->setEnabled(false);
    p_peakFindAct->setEnabled(false);
    p_overlayAct->setEnabled(false);

    // Connect curveMetadataChanged signal from all FT plots
    QList<FtPlot*> ftPlots = findChildren<FtPlot*>();
    for (FtPlot* plot : ftPlots) {
        if (plot)
            connect(plot, &ZoomPanPlot::curveMetadataChanged, this, &FtmwViewWidget::onCurveMetadataChanged);
    }

    createPlotNamesList();

    restoreDockLayout();
}

void FtmwViewWidget::setupInnerUi()
{
    p_innerWindow = new QMainWindow(this);
    p_innerWindow->setWindowFlags(Qt::Widget);
    p_innerWindow->setDockOptions(QMainWindow::AnimatedDocks
                                  | QMainWindow::AllowTabbedDocks
                                  | QMainWindow::AllowNestedDocks);
    p_innerWindow->setTabPosition(Qt::AllDockWidgetAreas, QTabWidget::North);
    p_innerWindow->setDockNestingEnabled(true);

    setupTopToolbar();

    // ── Central widget: experiment label header + splitter ─────────────
    auto *central = new QWidget(p_innerWindow);
    auto *centralLay = new QVBoxLayout(central);
    centralLay->setContentsMargins(0,0,0,0);

    p_exptLabel = new QLabel(central);
    QFont boldFont;
    boldFont.setBold(true);
    p_exptLabel->setFont(boldFont);
    p_exptLabel->setAlignment(Qt::AlignCenter);
    p_exptLabel->setText("Experiment");
    centralLay->addWidget(p_exptLabel,0);

    p_splitter = new QSplitter(Qt::Vertical, central);
    p_splitter->setChildrenCollapsible(false);
    centralLay->addWidget(p_splitter,1);

    p_topPlotsContainer = new QWidget(p_splitter);
    auto *topLayout = new QVBoxLayout(p_topPlotsContainer);
    topLayout->setContentsMargins(0,0,0,0);

    p_liveRowWidget = new QWidget(p_topPlotsContainer);
    auto *liveRow = new QHBoxLayout(p_liveRowWidget);
    liveRow->setContentsMargins(0,0,0,0);
    p_liveFidPlot = new FidPlot(QString("Live"),p_liveRowWidget);
    p_liveFidPlot->setObjectName("liveFidPlot");
    p_liveFtPlot = new FtPlot(QString("Live"),p_liveRowWidget);
    p_liveFtPlot->setObjectName("liveFtPlot");
    liveRow->addWidget(p_liveFidPlot);
    liveRow->addWidget(p_liveFtPlot);
    liveRow->setStretch(0,1);
    liveRow->setStretch(1,1);
    topLayout->addWidget(p_liveRowWidget,1);

    p_plot12RowWidget = new QWidget(p_topPlotsContainer);
    auto *plot12Row = new QHBoxLayout(p_plot12RowWidget);
    plot12Row->setContentsMargins(0,0,0,0);
    p_fidPlot1 = new FidPlot(QString("1"),p_plot12RowWidget);
    p_fidPlot1->setObjectName("fidPlot1");
    p_ftPlot1 = new FtPlot(QString("1"),p_plot12RowWidget);
    p_ftPlot1->setObjectName("ftPlot1");
    plot12Row->addWidget(p_fidPlot1);
    plot12Row->addWidget(p_ftPlot1);
    p_fidPlot2 = new FidPlot(QString("2"),p_plot12RowWidget);
    p_fidPlot2->setObjectName("fidPlot2");
    p_ftPlot2 = new FtPlot(QString("2"),p_plot12RowWidget);
    p_ftPlot2->setObjectName("ftPlot2");
    plot12Row->addWidget(p_fidPlot2);
    plot12Row->addWidget(p_ftPlot2);
    topLayout->addWidget(p_plot12RowWidget,1);

    p_splitter->addWidget(p_topPlotsContainer);

    p_mainFtPlot = new MainFtPlot(p_splitter);
    p_mainFtPlot->setObjectName("mainFtPlot");
    p_mainFtPlot->setMinimumSize(QSize(0,100));
    p_splitter->addWidget(p_mainFtPlot);
    p_splitter->setStretchFactor(0,1);
    p_splitter->setStretchFactor(1,2);

    p_innerWindow->setCentralWidget(central);

    // ── Side-panel docks ───────────────────────────────────────────────
    p_processingPanel = new FtmwProcessingPanel(d_main, p_innerWindow);
    p_plotPanel = new FtmwPlotPanel(p_innerWindow);
    if(d_main)
        p_acquisitionPanel = new FtmwAcquisitionPanel(d_main, p_innerWindow);

    p_processingDock  = makeDock("FtmwViewDock-Processing",  "FID Processing", p_processingPanel);
    p_plotDock        = makeDock("FtmwViewDock-Plot",        "Plot Settings",  p_plotPanel);
    if(d_main)
        p_acquisitionDock = makeDock("FtmwViewDock-Acquisition", "Acquisition", p_acquisitionPanel);
    p_peakFindDock    = makeDock("FtmwViewDock-PeakFind",    "Peak Find",      nullptr);
    p_overlayDock     = makeDock("FtmwViewDock-Overlays",    "Overlays",       nullptr);

    // Build toolbar toggle actions from the docks' toggleViewAction so
    // checked state stays in sync with dock visibility automatically.
    p_processingAct = p_processingDock->toggleViewAction();
    p_plotAct       = p_plotDock->toggleViewAction();
    if(p_acquisitionDock)
        p_acquisitionAct = p_acquisitionDock->toggleViewAction();
    p_peakFindAct   = p_peakFindDock->toggleViewAction();
    p_overlayAct    = p_overlayDock->toggleViewAction();

    p_processingAct->setIcon(ThemeColors::createThemedIcon(":/icons/presentation-chart-line.svg",ThemeColors::IconPrimary,this));
    p_processingAct->setText("FID Processing");
    p_plotAct->setIcon(ThemeColors::createThemedIcon(":/icons/presentation-chart-bar.svg",ThemeColors::IconSecondary,this));
    p_plotAct->setText("Plot Settings");
    if(p_acquisitionAct)
    {
        p_acquisitionAct->setIcon(ThemeColors::createThemedIcon(":/icons/arrow-trending-up.svg",ThemeColors::IconSecondary,this));
        p_acquisitionAct->setText("Acquisition");
    }
    p_peakFindAct->setIcon(ThemeColors::createThemedIcon(":/icons/magnifying-glass-circle.svg",ThemeColors::IconPrimary,this));
    p_peakFindAct->setText("Peak Find");
    p_overlayAct->setIcon(ThemeColors::createThemedIcon(":/icons/squares-plus.svg",ThemeColors::IconSecondary,this));
    p_overlayAct->setText("Overlays");

    p_topToolbar->addAction(p_processingAct);
    p_topToolbar->addAction(p_plotAct);
    if(p_acquisitionAct)
        p_topToolbar->addAction(p_acquisitionAct);
    p_topToolbar->addAction(p_peakFindAct);
    p_topToolbar->addAction(p_overlayAct);

    p_innerWindow->addDockWidget(Qt::RightDockWidgetArea, p_processingDock);
    p_innerWindow->addDockWidget(Qt::RightDockWidgetArea, p_plotDock);
    if(p_acquisitionDock)
        p_innerWindow->addDockWidget(Qt::RightDockWidgetArea, p_acquisitionDock);
    p_innerWindow->addDockWidget(Qt::RightDockWidgetArea, p_peakFindDock);
    p_innerWindow->addDockWidget(Qt::RightDockWidgetArea, p_overlayDock);

    // Default to all hidden (collapsed); restoreDockLayout() may override.
    p_processingDock->setVisible(false);
    p_plotDock->setVisible(false);
    if(p_acquisitionDock)
        p_acquisitionDock->setVisible(false);
    p_peakFindDock->setVisible(false);
    p_overlayDock->setVisible(false);

    // Lazy-construct PeakFind / Overlay widgets when their dock is first
    // shown and a suitable Ft / experiment is available.
    connect(p_peakFindDock,&QDockWidget::visibilityChanged,this,&FtmwViewWidget::showPeakFinder);
    connect(p_overlayDock,&QDockWidget::visibilityChanged,this,&FtmwViewWidget::showOverlayManager);

    // Outer layout: just hosts the inner main window
    auto *vbl = new QVBoxLayout;
    vbl->setContentsMargins(0,0,0,0);
    vbl->addWidget(p_innerWindow);
    setLayout(vbl);

    resize(850,520);
}

void FtmwViewWidget::setupTopToolbar()
{
    p_topToolbar = new QToolBar(p_innerWindow);
    p_topToolbar->setObjectName("FtmwViewTopToolbar");
    p_topToolbar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    p_topToolbar->setMovable(false);
    p_topToolbar->setFloatable(false);
    p_innerWindow->addToolBar(Qt::TopToolBarArea, p_topToolbar);
}

QDockWidget *FtmwViewWidget::makeDock(const QString &objectName, const QString &title, QWidget *contents)
{
    auto *dock = new QDockWidget(title, p_innerWindow);
    dock->setObjectName(objectName);
    dock->setFeatures(QDockWidget::DockWidgetMovable
                      | QDockWidget::DockWidgetFloatable
                      | QDockWidget::DockWidgetClosable);
    dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    if(contents)
        dock->setWidget(contents);
    else
        resetDockToPlaceholder(dock,
                               QString("%1 panel will appear when an experiment is loaded.").arg(title));
    return dock;
}

void FtmwViewWidget::resetDockToPlaceholder(QDockWidget *dock, const QString &message)
{
    if (!dock)
        return;
    QWidget *prev = dock->widget();
    auto *placeholder = new QLabel(message);
    placeholder->setAlignment(Qt::AlignCenter);
    placeholder->setWordWrap(true);
    placeholder->setMargin(12);
    dock->setWidget(placeholder);  // unparents `prev`; dock takes ownership of placeholder
    if (prev)
        prev->deleteLater();
}

QLatin1StringView FtmwViewWidget::dockStateKey() const
{
    return d_main ? BC::Key::FtmwView::dockStateMain
                  : BC::Key::FtmwView::dockStateViewer;
}

QByteArray FtmwViewWidget::defaultDockStateBlob() const
{
    // Empty blob: restoreState() is a no-op and the widget keeps the
    // post-construction default (all docks tabified on the right edge,
    // all hidden).
    return {};
}

void FtmwViewWidget::restoreDockLayout()
{
    auto blob = get<QByteArray>(dockStateKey(), defaultDockStateBlob());
    if(!blob.isEmpty())
        p_innerWindow->restoreState(blob);
}

void FtmwViewWidget::persistDockLayout()
{
    set(dockStateKey(), p_innerWindow->saveState(), false);
}

FtmwViewWidget::~FtmwViewWidget()
{
    // Tear down the lazy-built dock contents (PeakFind, OverlayManager)
    // first: detach signals, drop them via the dock helper, and let the
    // event loop reap them after we return.
    closeOverlayManager();

    if(p_pfw != nullptr)
    {
        disconnect(p_worker, nullptr, p_pfw, nullptr);
        disconnect(p_pfw, nullptr, p_mainFtPlot, nullptr);
        p_pfw = nullptr;
        resetDockToPlaceholder(p_peakFindDock,
            "Peak Find panel will appear when an experiment is loaded.");
    }

    persistDockLayout();
    clearGetters();

    saveOverlays();

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

    if (ps_overlayStorage)
        ps_overlayStorage->waitForPendingWrites();
}

void FtmwViewWidget::prepareForExperiment(const Experiment &e)
{
    if(p_pfw != nullptr)
    {
        // Disconnect everything before unparenting + scheduling deletion to
        // ensure no signal fires on a half-detached widget.
        disconnect(p_worker, nullptr, p_pfw, nullptr);
        disconnect(p_pfw, nullptr, p_mainFtPlot, nullptr);
        p_pfw = nullptr;
        resetDockToPlaceholder(p_peakFindDock,
            "Peak Find panel will appear when an experiment is loaded.");
    }

    closeOverlayManager();
    if(p_overlayDock && p_overlayDock->widget() == nullptr)
        resetDockToPlaceholder(p_overlayDock,
            "Overlays panel will appear when an experiment is loaded.");

    if(!p_exptLabel->isVisible())
        p_exptLabel->setVisible(true);

    if(p_acquisitionPanel)
        p_acquisitionPanel->setRefreshEnabled(false);

    p_liveFidPlot->prepareForExperiment(e);
    p_liveFidPlot->setVisible(true);

    p_fidPlot1->prepareForExperiment(e);
    p_fidPlot2->prepareForExperiment(e);

    p_liveFtPlot->prepareForExperiment(e);
    p_processingPanel->prepareForExperient(e);
    p_plotPanel->prepareForExperiment(e);
    p_ftPlot1->prepareForExperiment(e);
    p_ftPlot2->prepareForExperiment(e);
    p_mainFtPlot->prepareForExperiment(e);

    d_currentSegment = 0;
    for(auto &[key,ps] : d_plotStatus)
    {
        Q_UNUSED(key)
        ps.fidList.clear();
        ps.ft = Ft();
        ps.frame = (key == d_liveId ? -1 : 0);
        ps.segment = 0;
        ps.backup = 0;
        ps.loadWhenDone = false;
    }

    if (ps_overlayStorage) {
        ps_overlayStorage->waitForPendingWrites();
        disconnect(ps_overlayStorage.get(), nullptr, this, nullptr);
        ps_overlayStorage->save();
    }

    ps_overlayStorage = e.overlayStorage();

    if (ps_overlayStorage) {
        connect(ps_overlayStorage.get(), &OverlayStorage::overlayAdded, this, &FtmwViewWidget::onOverlayAdded);
        connect(ps_overlayStorage.get(), &OverlayStorage::overlayRemoved, this, &FtmwViewWidget::onOverlayRemoved);
    }

    if (ps_overlayStorage && d_overlaysEnabled)
    {
        auto overlays = ps_overlayStorage->getAllOverlays();
        for (const auto& overlay : overlays)
            addOverlayToPlots(overlay);
    }

    if (ps_overlayStorage && d_overlaysEnabled) {
        for (const auto &overlay : d_overlaysToCopy) {
            ps_overlayStorage->addOverlay(overlay);
        }
    }
    d_overlaysToCopy.clear();

    if(e.ftmwEnabled())
    {
        if(p_acquisitionPanel)
        {
            p_acquisitionPanel->setRefreshEnabled(true);
            connect(p_acquisitionPanel,&FtmwAcquisitionPanel::refreshIntervalChanged,
                    this,&FtmwViewWidget::setLiveUpdateInterval, Qt::UniqueConnection);
        }
        ps_fidStorage = e.ftmwConfig()->storage();
        if(e.ftmwConfig()->d_type == FtmwConfig::Peak_Up)
            p_exptLabel->setText(QString("Peak Up Mode"));
        else
            p_exptLabel->setText(QString("Experiment %1").arg(e.d_number));

        d_currentExptNum = e.d_number;

        p_liveRowWidget->setVisible(true);
        p_liveFidPlot->show();
        p_liveFtPlot->show();

        if(p_acquisitionPanel)
        {
            const bool isPeakUp = (e.ftmwConfig()->d_type == FtmwConfig::Peak_Up);
            p_acquisitionPanel->setAverages(isPeakUp ? e.ftmwConfig()->d_objective : 0);
            p_acquisitionPanel->setPeakUpControlsEnabled(isPeakUp);

            // Manual backup is meaningful only for single-segment, backup-capable
            // FTMW modes. Hidden entirely in the viewer (constructor of panel).
            const auto t = e.ftmwConfig()->d_type;
            p_acquisitionPanel->setManualBackupEnabled(
                t == FtmwConfig::Target_Shots
                || t == FtmwConfig::Target_Duration
                || t == FtmwConfig::Forever);

            d_liveTimerId = startTimer(p_acquisitionPanel->refreshInterval());
        }
        else
        {
            d_liveTimerId = startTimer(get(BC::Key::FtmwView::refresh,500));
        }
    }
    else
    {
        ps_fidStorage.reset();
        ps_overlayStorage.reset();
        p_exptLabel->setText(QString("Experiment %1").arg(e.d_number));
        if(p_acquisitionPanel)
        {
            p_acquisitionPanel->setPeakUpControlsEnabled(false);
            p_acquisitionPanel->setManualBackupEnabled(false);
        }
    }

    p_peakFindAct->setEnabled(false);
    p_overlayAct->setEnabled(false);
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
                if(!p_plotPanel->viewingBackup(key))
                {
                    ps.fidList = fl;
                    process(key,fl,ps.frame);
                }
                else if(p_plotPanel->differential(key))
                {
                    updateFid(key);
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
    d_currentProcessingSettings = s;
    QList<int> ignore;

    if(!p_liveFidPlot->isHidden())
    {
        p_liveFidPlot->setFtStart(s.startUs);
        p_liveFidPlot->setFtEnd(s.endUs);
    }
    else
        ignore << d_liveId;

    p_fidPlot1->setFtStart(s.startUs);
    p_fidPlot1->setFtEnd(s.endUs);
    p_fidPlot2->setFtStart(s.startUs);
    p_fidPlot2->setFtEnd(s.endUs);

    reprocess(ignore);
}

void FtmwViewWidget::resetProcessingSettings()
{
    if(ps_fidStorage)
    {
        if(ps_fidStorage->readProcessingSettings(d_currentProcessingSettings))
            p_processingPanel->setAll(d_currentProcessingSettings);
    }
}

void FtmwViewWidget::saveProcessingSettings()
{
    if(ps_fidStorage)
        ps_fidStorage->writeProcessingSettings(p_processingPanel->getSettings());
}

void FtmwViewWidget::updatePlotSetting(int id)
{
    auto it = d_plotStatus.find(id);
    if(it != d_plotStatus.end())
    {
        //segment and frame are 1-indexed on the UI
        it->second.segment = p_plotPanel->segment(id)-1;
        it->second.frame = p_plotPanel->frame(id)-1;
        it->second.backup = p_plotPanel->backup(id);
        it->second.differential = p_plotPanel->differential(id);
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
    if(ws.reprocessWhenDone)
    {
        if(id == d_mainId)
        {
            switch(p_plotPanel->mainPlotMode())
            {
            case FtmwPlotPanel::Lower_SideBand:
            case FtmwPlotPanel::Upper_SideBand:
            case FtmwPlotPanel::Both_SideBands:
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

        switch(p_plotPanel->mainPlotMode()) {
        case FtmwPlotPanel::Live:
        case FtmwPlotPanel::FT1:
        case FtmwPlotPanel::FT2:
        case FtmwPlotPanel::FT1_minus_FT2:
        case FtmwPlotPanel::FT2_minus_FT1:
            updateMainPlot();
            break;
        default:
            break;
        }
    }
    else
    {
        //this is the main plot
        p_mainFtPlot->newFt(ft);
        p_peakFindAct->setEnabled(!ft.isEmpty());
        p_overlayAct->setEnabled(!ft.isEmpty() && d_overlaysEnabled);
        p_mainFtPlot->canvas()->setCursor(QCursor(Qt::CrossCursor));
        if(p_pfw != nullptr)
            p_pfw->newFt(ft);
    }
}

void FtmwViewWidget::ftDiffDone(const Ft ft)
{
    p_mainFtPlot->newFt(ft);
    p_mainFtPlot->canvas()->setCursor(QCursor(Qt::CrossCursor));
}

void FtmwViewWidget::updateMainPlot()
{
    p_mainFtPlot->configureUnits(d_currentProcessingSettings.units);
    p_mainFtPlot->setMessageText("");
    p_mainFtPlot->canvas()->setCursor(QCursor(Qt::CrossCursor));

    cancelSidebandProcessing();

    switch(p_plotPanel->mainPlotMode()) {
    case FtmwPlotPanel::Live:
        p_mainFtPlot->newFt(d_plotStatus[d_liveId].ft);
        if(p_pfw != nullptr)
            p_pfw->newFt(d_plotStatus[d_liveId].ft);
        break;
    case FtmwPlotPanel::FT1:
        p_mainFtPlot->newFt(d_plotStatus[d_plot1Id].ft);
        if(p_pfw != nullptr)
            p_pfw->newFt(d_plotStatus[d_plot1Id].ft);
        break;
    case FtmwPlotPanel::FT2:
        p_mainFtPlot->newFt(d_plotStatus[d_plot2Id].ft);
        if(p_pfw != nullptr)
            p_pfw->newFt(d_plotStatus[d_plot2Id].ft);
        break;
    case FtmwPlotPanel::FT1_minus_FT2:
        processDiff(d_plotStatus[d_plot1Id].fidList,d_plotStatus[d_plot2Id].fidList,
                    d_plotStatus[d_plot1Id].frame,d_plotStatus[d_plot2Id].frame);
        break;
    case FtmwPlotPanel::FT2_minus_FT1:
        processDiff(d_plotStatus[d_plot2Id].fidList,d_plotStatus[d_plot1Id].fidList,
                    d_plotStatus[d_plot2Id].frame,d_plotStatus[d_plot1Id].frame);
        break;
    case FtmwPlotPanel::Upper_SideBand:
    case FtmwPlotPanel::Lower_SideBand:
    case FtmwPlotPanel::Both_SideBands:
        processSidebands();
        break;
    }

    p_peakFindAct->setEnabled(!p_mainFtPlot->currentFt().isEmpty());
    p_overlayAct->setEnabled(!p_mainFtPlot->currentFt().isEmpty() && d_overlaysEnabled);
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
        p_mainFtPlot->canvas()->setCursor(QCursor(Qt::BusyCursor));
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
        sbd.frame = p_plotPanel->sbFrame()-1;
        sbd.minOffset = p_plotPanel->sbMinFreq();
        sbd.maxOffset = p_plotPanel->sbMaxFreq();
        sbd.dcMethod = p_plotPanel->dcMethod();
        sbd.totalFids = storage->numSegments();
        sbd.loRange = storage->getLORange();
        switch (p_plotPanel->mainPlotMode()) {
        case FtmwPlotPanel::Lower_SideBand:
            sbd.doubleSideband = false;
            sbd.sideband = RfConfig::LowerSideband;
            break;
        case FtmwPlotPanel::Upper_SideBand:
            sbd.doubleSideband = false;
            sbd.sideband = RfConfig::UpperSideband;
            break;
        case FtmwPlotPanel::Both_SideBands:
            sbd.doubleSideband  = true;
            break;
        default:
            break;
        };
        d_sbStatus.sbData = sbd;

        p_mainFtPlot->canvas()->setCursor(QCursor(Qt::BusyCursor));
        p_mainFtPlot->setMessageText(QString("Processing..."));
        p_mainFtPlot->newFt(Ft());
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
        p_worker->processSideband(sbd,d_currentProcessingSettings);
    }));
    d_sbStatus.sbData.currentIndex++;
    p_mainFtPlot->setMessageText(QString("Processing %1/%2")
                                 .arg(d_sbStatus.sbData.currentIndex)
                                 .arg(d_sbStatus.sbData.totalFids));
    p_mainFtPlot->replot();
}

void FtmwViewWidget::sidebandProcessingComplete(const Ft ft)
{
    d_sbStatus.complete = true;

    if(d_sbStatus.cancel)
        updateMainPlot();
    else
    {
        d_sbStatus.nextFidList = FidList();

        p_mainFtPlot->canvas()->setCursor(QCursor(Qt::CrossCursor));
        p_mainFtPlot->setMessageText("");
        p_mainFtPlot->newFt(ft);
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

    p_plotPanel->newBackup(ps_fidStorage->numBackups());

    // Re-enable the manual backup button now that the write is on disk.
    if(p_acquisitionPanel && dynamic_cast<FidSingleStorage*>(ps_fidStorage.get()) != nullptr)
        p_acquisitionPanel->setManualBackupEnabled(true);
}

void FtmwViewWidget::experimentComplete()
{
    if(p_acquisitionPanel)
    {
        disconnect(p_acquisitionPanel,&FtmwAcquisitionPanel::refreshIntervalChanged,
                   this,&FtmwViewWidget::setLiveUpdateInterval);
        p_acquisitionPanel->setRefreshEnabled(false);
    }
    if(d_liveTimerId >= 0)
        killTimer(d_liveTimerId);
    d_liveTimerId = -1;

    if(p_acquisitionPanel)
        p_acquisitionPanel->setManualBackupEnabled(false);

    if(ps_fidStorage)
    {
        d_currentSegment = -1;

        p_liveRowWidget->setVisible(false);
        p_liveFidPlot->hide();
        p_liveFtPlot->hide();

        p_plotPanel->experimentComplete();

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

void FtmwViewWidget::showPeakFinder(bool show)
{
    if(!show)
        return;

    if(p_pfw != nullptr || p_mainFtPlot->currentFt().isEmpty())
        return;

    p_pfw = new PeakFindWidget(p_mainFtPlot->currentFt(),d_currentExptNum);
    p_peakFindDock->setWindowTitle(d_currentExptNum > 0
                                   ? QString("Peak List: Experiment %1").arg(d_currentExptNum)
                                   : QString("Peak List"));

    // Backstop against external deletion (matches showOverlayManager).
    connect(p_pfw, &QObject::destroyed, this, [this](QObject *obj){
        if (obj == p_pfw) p_pfw = nullptr;
    });

    connect(p_worker,&FtWorker::ftDone,p_pfw,[this](const Ft ft, int id){
        if(id == d_mainId && p_pfw)
            p_pfw->newFt(ft);
    });
    connect(p_pfw,&PeakFindWidget::peakList,p_mainFtPlot,&MainFtPlot::newPeakList);
    connect(p_pfw,&PeakFindWidget::editPeakAppearanceRequested,
            p_mainFtPlot,&MainFtPlot::showPeakAppearanceMenu);
    connect(p_mainFtPlot,&ZoomPanPlot::visibleXRangeChanged,
            p_pfw,&PeakFindWidget::setMainPlotXRange);
    // Seed the current range so the in-view filter is correct before the
    // next replot emits.
    const auto xiv = p_mainFtPlot->axisInterval(QwtPlot::xBottom);
    if(xiv.isValid())
        p_pfw->setMainPlotXRange(xiv.minValue(),xiv.maxValue());

    p_peakFindDock->setWidget(p_pfw);
}

void FtmwViewWidget::showOverlayManager(bool show)
{
    if(!show)
        return;

    if(p_omw != nullptr || !ps_overlayStorage)
        return;

    p_omw = new OverlayManagerWidget(p_overlayDock, d_currentExptNum, getAllOverlays());

    // Backstop: if the widget is ever destroyed by something other than
    // closeOverlayManager(), null our pointer so we don't dereference it.
    connect(p_omw, &QObject::destroyed, this, [this](QObject *obj){
        if (obj == p_omw) p_omw = nullptr;
    });

    if (ps_overlayStorage)
        p_omw->connectToOverlayStorage(ps_overlayStorage);

    connect(p_omw, &OverlayManagerWidget::overlayDataChanged,
            this, &FtmwViewWidget::onOverlayDataChanged);
    connect(this, &FtmwViewWidget::externalOverlayDataChanged,
            p_omw, &OverlayManagerWidget::onExternalOverlayDataChanged);

    p_overlayDock->setWidget(p_omw);
}

void FtmwViewWidget::updateFid(int id)
{
    if(id == d_mainId)
        return;

    auto &ps = d_plotStatus[id];
    int seg = ps.segment;
    int backup = ps.backup;
    bool diff = ps.differential;

    if(seg == d_currentSegment && id == d_liveId)
    {
        auto fl = ps_fidStorage->getCurrentFidList();
        ps.fidList = fl;
        process(id, ps.fidList, -1);
    }
    else
    {
        if(backup > 0)
            seg = backup;

        if(ps.p_watcher->isRunning())
            ps.loadWhenDone = true;
        else
        {
            if(diff && backup > 0)
                ps.p_watcher->setFuture(QtConcurrent::run([this,seg](){ return ps_fidStorage->loadDifferentialFidList(seg); }));
            else
                ps.p_watcher->setFuture(QtConcurrent::run([this,seg](){ return ps_fidStorage->loadFidList(seg); }));
        }
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

QVector<std::shared_ptr<OverlayBase>> FtmwViewWidget::getAllOverlays() const
{
    if (ps_overlayStorage)
        return ps_overlayStorage->getAllOverlays();
    return QVector<std::shared_ptr<OverlayBase>>();
}

void FtmwViewWidget::addOverlay(std::shared_ptr<OverlayBase> overlay)
{
    if(overlay != nullptr) {
        if (ps_overlayStorage) {
            bool added = ps_overlayStorage->addOverlay(overlay);
            if (!added) {
                return;
            }
        }
        addOverlayToPlots(overlay);
    }
}

void FtmwViewWidget::removeOverlay(std::shared_ptr<OverlayBase> overlay)
{
    if(overlay != nullptr && ps_overlayStorage) {
        ps_overlayStorage->removeOverlay(overlay->getLabel());
        removeOverlayFromPlots(overlay);
    }
}

void FtmwViewWidget::addOverlayToPlots(std::shared_ptr<OverlayBase> overlay)
{
    if (!overlay)
        return;

    QString plotName = overlay->getPlotId();
    auto it = d_plotMap.find(plotName);
    if (it != d_plotMap.end())
        it->second->addOverlay(overlay);
}

void FtmwViewWidget::onOverlayAdded(std::shared_ptr<OverlayBase> overlay)
{
    addOverlayToPlots(overlay);
}

void FtmwViewWidget::removeOverlayFromPlots(std::shared_ptr<OverlayBase> overlay)
{
    if (!overlay)
        return;

    QString plotName = overlay->getPlotId();
    auto it = d_plotMap.find(plotName);
    if (it != d_plotMap.end())
        it->second->removeOverlay(overlay);
}

void FtmwViewWidget::onOverlayRemoved(std::shared_ptr<OverlayBase> overlay)
{
    removeOverlayFromPlots(overlay);
}

void FtmwViewWidget::onOverlayDataChanged(std::shared_ptr<OverlayBase> overlay)
{
    if (!overlay)
        return;

    QString targetPlotName = overlay->getPlotId();

    QString currentPlotName;
    for (auto& [plotName, plot] : d_plotMap) {
        if (plot->hasOverlay(overlay)) {
            currentPlotName = plotName;
            break;
        }
    }

    if (!currentPlotName.isEmpty() && currentPlotName != targetPlotName) {
        auto currentIt = d_plotMap.find(currentPlotName);
        if (currentIt != d_plotMap.end())
            currentIt->second->removeOverlay(overlay);

        auto targetIt = d_plotMap.find(targetPlotName);
        if (targetIt != d_plotMap.end())
            targetIt->second->addOverlay(overlay);
    }
    else if (currentPlotName.isEmpty()) {
        auto targetIt = d_plotMap.find(targetPlotName);
        if (targetIt != d_plotMap.end())
            targetIt->second->addOverlay(overlay);
        else
            qWarning() << "Target plot" << targetPlotName << "not found!";
    }
    else {
        auto targetIt = d_plotMap.find(targetPlotName);
        if (targetIt != d_plotMap.end())
            targetIt->second->updateOverlay(overlay);
        else
            qWarning() << "Target plot" << targetPlotName << "not found!";
    }
}

void FtmwViewWidget::onCurveMetadataChanged(BlackchirpPlotCurveBase* curve)
{
    if (!curve || !ps_overlayStorage)
        return;

    if (curve->getStorageType() == BlackchirpPlotCurveBase::StorageType::OverlayMetadata) {
        auto overlay = curve->getOverlay();
        if (overlay)
            ps_overlayStorage->saveOverlayMetadata(overlay);
    }
}

void FtmwViewWidget::createPlotNamesList()
{
    d_plotNames.clear();
    d_plotMap.clear();

    QList<FtPlot*> ftPlots = findChildren<FtPlot*>();

    for(FtPlot* plot : ftPlots) {
        if(plot != nullptr) {
            QString plotName = plot->objectName();
            if(!plotName.isEmpty()) {
                QString nameLower = plotName.toLower();
                if(nameLower.contains("ft") && nameLower.contains("live"))
                    continue;
                d_plotNames.append(plotName);
                d_plotMap[plotName] = plot;

                connect(plot, &FtPlot::overlayDataChanged,
                        this, &FtmwViewWidget::onOverlayDataChanged);

                connect(plot, &FtPlot::overlayDataChanged,
                        this, &FtmwViewWidget::externalOverlayDataChanged);
            }
        }
    }

    d_plotNames.sort();
}

Ft FtmwViewWidget::getMainPlotFt() const
{
    if (p_mainFtPlot)
        return p_mainFtPlot->currentFt();
    return Ft();
}

void FtmwViewWidget::saveOverlays()
{
    if (ps_overlayStorage) {
        ps_overlayStorage->waitForPendingWrites();
        ps_overlayStorage->save();
    }
}

void FtmwViewWidget::closeOverlayManager()
{
    if (!p_omw)
        return;

    // Disconnect every connection we set up in either direction so no
    // signal can fire on a widget that is about to be destroyed.
    disconnect(p_omw, nullptr, this, nullptr);
    disconnect(this, nullptr, p_omw, nullptr);
    OverlayManagerWidget *doomed = p_omw;
    p_omw = nullptr;
    if (p_overlayDock && p_overlayDock->widget() == doomed)
        p_overlayDock->setWidget(nullptr);  // unparents `doomed`
    doomed->deleteLater();
}

bool FtmwViewWidget::promptOverlayTransition()
{
    closeOverlayManager();

    d_overlaysToCopy.clear();

    if (!ps_overlayStorage || !d_overlaysEnabled)
        return true;

    auto allOverlays = ps_overlayStorage->getAllOverlays();
    if (allOverlays.isEmpty())
        return true;

    QDialog dlg(this);
    dlg.setWindowTitle("Overlay Transition");

    auto *layout = new QVBoxLayout(&dlg);

    auto *label = new QLabel("Select overlays to copy to the new experiment:");
    layout->addWidget(label);

    auto *tree = new QTreeWidget;
    tree->setHeaderLabels({"Name", "Type"});
    tree->setRootIsDecorated(false);
    tree->setSelectionMode(QAbstractItemView::NoSelection);

    auto typeName = [](OverlayBase::OverlayType t) -> QString {
        switch (t) {
        case OverlayBase::BCExperiment: return "BC Experiment";
        case OverlayBase::Catalog:      return "Catalog";
        case OverlayBase::GenericXY:    return "Generic XY";
        }
        return "Unknown";
    };

    for (const auto &overlay : allOverlays) {
        auto *item = new QTreeWidgetItem(tree);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(0, Qt::Checked);
        item->setText(0, overlay->getLabel());
        item->setText(1, typeName(overlay->type()));
    }

    tree->resizeColumnToContents(0);
    tree->resizeColumnToContents(1);
    layout->addWidget(tree);

    auto *selectionLayout = new QHBoxLayout;
    auto *selectAllBtn = new QPushButton("Select All");
    auto *selectNoneBtn = new QPushButton("Select None");
    selectionLayout->addWidget(selectAllBtn);
    selectionLayout->addWidget(selectNoneBtn);
    selectionLayout->addStretch();
    layout->addLayout(selectionLayout);

    QObject::connect(selectAllBtn, &QPushButton::clicked, [tree]() {
        for (int i = 0; i < tree->topLevelItemCount(); ++i)
            tree->topLevelItem(i)->setCheckState(0, Qt::Checked);
    });
    QObject::connect(selectNoneBtn, &QPushButton::clicked, [tree]() {
        for (int i = 0; i < tree->topLevelItemCount(); ++i)
            tree->topLevelItem(i)->setCheckState(0, Qt::Unchecked);
    });

    auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    QObject::connect(buttonBox, &QDialogButtonBox::accepted, &dlg, &QDialog::accept);
    QObject::connect(buttonBox, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
    layout->addWidget(buttonBox);

    dlg.resize(400, 300);

    if (dlg.exec() != QDialog::Accepted)
        return false;

    for (int i = 0; i < tree->topLevelItemCount(); ++i) {
        if (tree->topLevelItem(i)->checkState(0) == Qt::Checked)
            d_overlaysToCopy.push_back(allOverlays.at(i));
    }

    return true;
}
