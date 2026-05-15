#ifndef FTMWVIEWWIDGET_H
#define FTMWVIEWWIDGET_H

#include <QWidget>
#include <QFutureWatcher>
#include <QList>
#include <map>
#include <memory>

#include <data/experiment/experiment.h>
#include <data/analysis/ftworker.h>
#include <data/storage/settingsstorage.h>
#include <data/experiment/overlaybase.h>
#include <data/storage/overlaystorage.h>

class QToolBar;
class QSplitter;
class QLabel;
class QMainWindow;
class QDockWidget;
class QAction;
class QThread;

class FidPlot;
class FtPlot;
class MainFtPlot;
class FtmwSnapshotWidget;
class PeakFindWidget;
class OverlayManagerWidget;
class BlackchirpPlotCurveBase;
class FtmwProcessingPanel;
class FtmwPlotPanel;
class FtmwAcquisitionPanel;

namespace BC::Key::FtmwView {
inline constexpr QLatin1StringView key{"FtmwViewWidget"};
inline constexpr QLatin1StringView refresh{"refreshMs"};
inline constexpr QLatin1StringView dockStateMain{"dockStateMain"};
inline constexpr QLatin1StringView dockStateViewer{"dockStateViewer"};
}

class FtmwViewWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    explicit FtmwViewWidget(bool main, QWidget *parent = nullptr, QString path = QString(""), bool overlaysEnabled = true);
    ~FtmwViewWidget();
    void prepareForExperiment(const Experiment &e);

    FtWorker::FidProcessingSettings getProcessingSettings() const { return d_currentProcessingSettings; }

    // Overlay management
    std::shared_ptr<OverlayStorage> getOverlayStorage() const { return ps_overlayStorage; }
    QVector<std::shared_ptr<OverlayBase>> getAllOverlays() const;
    void addOverlay(std::shared_ptr<OverlayBase> overlay);
    void removeOverlay(std::shared_ptr<OverlayBase> overlay);
    bool promptOverlayTransition();

    // Plot management
    QStringList getPlotNames() const { return d_plotNames; }
    Ft getMainPlotFt() const;

signals:
    void externalOverlayDataChanged(std::shared_ptr<OverlayBase> overlay);

    /// \brief Emitted when the user clicks the manual backup action.
    ///
    /// Connected to AcquisitionManager::requestBackup via a queued connection
    /// (cross-thread). The action is disabled until the next \c updateBackups()
    /// call so a single click cannot stack multiple requests.
    void manualBackupRequested();

public slots:
    void setLiveUpdateInterval(int intervalms);
    void updateLiveFidList();
    void updateProcessingSettings(FtWorker::FidProcessingSettings s);
    void resetProcessingSettings();
    void saveProcessingSettings();
    void updatePlotSetting(int id);

    void fidLoadComplete(int id);
    void ftProcessingComplete(int id);
    void fidProcessed(const QVector<double> fidData, double spacing, double min, double max, quint64 shots, int workerId);
    void ftDone(const Ft ft, int workerId);
    void ftDiffDone(const Ft ft);
    void updateMainPlot();
    void reprocess(const QList<int> ignore = QList<int>());
    void process(int id, const FidList fl, int frame=0);
    void processDiff(const FidList fl1, const FidList fl2, int frame1, int frame2);

    void sidebandLoadComplete();
    void processSidebands();
    void loadNextSidebandFid();
    void processNextSidebandFid();
    void sidebandProcessingComplete(const Ft ft);
    void cancelSidebandProcessing();

    void updateBackups();
    void experimentComplete();

    // Overlay display slots
    void onOverlayAdded(std::shared_ptr<OverlayBase> overlay);
    void onOverlayRemoved(std::shared_ptr<OverlayBase> overlay);
    void onOverlayDataChanged(std::shared_ptr<OverlayBase> overlay);
    void onCurveMetadataChanged(BlackchirpPlotCurveBase* curve);

    // Internal overlay display methods
    void addOverlayToPlots(std::shared_ptr<OverlayBase> overlay);
    void removeOverlayFromPlots(std::shared_ptr<OverlayBase> overlay);

    void changeRollingAverageShots(int shots);
    void resetRollingAverage();

    void showPeakFinder(bool show);
    void showOverlayManager(bool show);
    void saveOverlays();

private:
    bool d_main;
    bool d_overlaysEnabled{true};

    // Inner main window hosting the plot area (central) and dock panels
    QMainWindow *p_innerWindow{nullptr};
    QToolBar *p_topToolbar{nullptr};
    QLabel *p_exptLabel{nullptr};
    QSplitter *p_splitter{nullptr};

    // Plot widgets
    FidPlot *p_liveFidPlot{nullptr};
    FtPlot *p_liveFtPlot{nullptr};
    FidPlot *p_fidPlot1{nullptr};
    FtPlot *p_ftPlot1{nullptr};
    FidPlot *p_fidPlot2{nullptr};
    FtPlot *p_ftPlot2{nullptr};
    MainFtPlot *p_mainFtPlot{nullptr};
    QWidget *p_topPlotsContainer{nullptr};
    QWidget *p_liveRowWidget{nullptr};
    QWidget *p_plot12RowWidget{nullptr};

    // Side-panel docks and their contents
    QDockWidget *p_processingDock{nullptr};
    QDockWidget *p_plotDock{nullptr};
    QDockWidget *p_acquisitionDock{nullptr};
    QDockWidget *p_peakFindDock{nullptr};
    QDockWidget *p_overlayDock{nullptr};
    FtmwProcessingPanel *p_processingPanel{nullptr};
    FtmwPlotPanel *p_plotPanel{nullptr};
    FtmwAcquisitionPanel *p_acquisitionPanel{nullptr};

    // Toolbar actions (each toggles its associated dock visibility)
    QAction *p_processingAct{nullptr};
    QAction *p_plotAct{nullptr};
    QAction *p_acquisitionAct{nullptr};
    QAction *p_peakFindAct{nullptr};
    QAction *p_overlayAct{nullptr};

    void setupInnerUi();
    void setupTopToolbar();
    QDockWidget *makeDock(const QString &objectName, const QString &title, QWidget *contents);
    void resetDockToPlaceholder(QDockWidget *dock, const QString &message);
    void restoreDockLayout();
    void persistDockLayout();
    QLatin1StringView dockStateKey() const;
    QByteArray defaultDockStateBlob() const;

    std::shared_ptr<FidStorageBase> ps_fidStorage;
    std::shared_ptr<OverlayStorage> ps_overlayStorage;
    FtWorker* p_worker;

    FtWorker::FidProcessingSettings d_currentProcessingSettings;
    int d_currentExptNum;
    int d_currentSegment;
    int d_liveTimerId{-1};

    struct WorkerStatus {
        QFutureWatcher<void> *p_watcher;
        bool busy;
        bool reprocessWhenDone;
    };

    struct PlotStatus {
        QFutureWatcher<FidList>* p_watcher;
        FidPlot *fidPlot;
        FtPlot *ftPlot;
        FidList fidList;
        Ft ft;
        int frame{0}; //only used for plot1 and plot2
        int segment{0}; //only used for plot1 and plot2
        int backup{0}; //only used for plot1 and plot2
        bool differential{false}; //only used for plot1 and plot2
        bool loadWhenDone{false};
    };

    QList<int> d_workerIds;
    std::map<int,WorkerStatus> d_workersStatus;
    std::map<int,PlotStatus> d_plotStatus;
    PeakFindWidget *p_pfw{nullptr};
    OverlayManagerWidget *p_omw{nullptr};
    QString d_path;
    QVector<std::shared_ptr<OverlayBase>> d_overlaysToCopy;
    QStringList d_plotNames;
    std::map<QString, FtPlot*, std::less<>> d_plotMap;
    const int d_liveId = 0, d_mainId = 3, d_plot1Id = 1, d_plot2Id = 2;
    const QString d_shotsString = QString("Shots: %1");

    struct SidebandStatus {
        QFutureWatcher<FidList> *sbLoadWatcher;
        FtWorker::SidebandProcessingData sbData;
        FidList nextFidList;
        bool cancel{true};
        bool complete{false};
    } d_sbStatus;


    void updateFid(int id);
    void createPlotNamesList();
    void closeOverlayManager();


    // QObject interface
protected:
    void timerEvent(QTimerEvent *event) override;
};

#endif // FTMWVIEWWIDGET_H
