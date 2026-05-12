#ifndef FTMWVIEWWIDGET_H
#define FTMWVIEWWIDGET_H

#include <climits>

#include <QWidget>
#include <QtCore/QVariant>
#include <QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QMenu>
#include <QtWidgets/QWidgetAction>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QFutureWatcher>
#include <QList>
#include <memory>

#include <data/experiment/experiment.h>
#include <data/analysis/ftworker.h>
#include <data/storage/settingsstorage.h>
#include <gui/plot/fidplot.h>
#include <gui/plot/ftplot.h>
#include <gui/plot/mainftplot.h>
#include <gui/widget/ftmwprocessingtoolbar.h>
#include <gui/widget/ftmwplottoolbar.h>
#include <gui/widget/toolbarwidgetaction.h>
#include <data/experiment/overlaybase.h>
#include <data/storage/overlaystorage.h>

class QThread;
class FtmwSnapshotWidget;
class PeakFindWidget;
class OverlayManagerWidget;
class BlackchirpPlotCurveBase;

namespace Ui {
class FtmwViewWidget;
}

namespace BC::Key::FtmwView {
inline constexpr QLatin1StringView key{"FtmwViewWidget"};
inline constexpr QLatin1StringView refresh{"refreshMs"};
}

class FtmwViewWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    explicit FtmwViewWidget(bool main, QWidget *parent = 0, QString path = QString(""), bool overlaysEnabled = true);
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

    /// \brief Emitted when the user clicks the manual backup toolbar action.
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

    void launchPeakFinder();
    void launchOverlayManager();
    void saveOverlays();


private:
    Ui::FtmwViewWidget *ui;
    
    void setupThemedIcons();

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
    // Overlay storage is now handled via ps_overlayStorage shared pointer
    QVector<std::shared_ptr<OverlayBase>> d_overlaysToCopy;
    QStringList d_plotNames;
    bool d_overlaysEnabled{true};
    std::map<QString, FtPlot*, std::less<>> d_plotMap;  // Maps plot names to FtPlot instances
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

QT_BEGIN_NAMESPACE

class Ui_FtmwViewWidget
{
public:
    QLabel *exptLabel;
    QSplitter *splitter;
    QWidget *widget;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *livePlotLayout;
    FidPlot *liveFidPlot;
    FtPlot *liveFtPlot;
    QHBoxLayout *plots12Layout;
    QHBoxLayout *plot1Layout;
    FidPlot *fidPlot1;
    FtPlot *ftPlot1;
    QHBoxLayout *plot2Layout;
    FidPlot *fidPlot2;
    FtPlot *ftPlot2;
    MainFtPlot *mainFtPlot;
    QToolBar *toolBar;
    QAction *processingAct;
    FtmwProcessingToolBar *processingToolBar;
    QAction *plotAction;
    FtmwPlotToolBar *plotToolBar;
    QSpinBox *averagesSpinbox;
    QPushButton *resetAveragesButton;
    QAction *peakFindAction;
    QAction *overlayAction;
    QAction *manualBackupAction;
    SpinBoxWidgetAction *refreshBox;

    void setupUi(bool main, QWidget *FtmwViewWidget)
    {
        if (FtmwViewWidget->objectName().isEmpty())
            FtmwViewWidget->setObjectName("FtmwViewWidget"_L1);
        FtmwViewWidget->resize(850, 520);
        exptLabel = new QLabel(FtmwViewWidget);
        exptLabel->setObjectName("exptLabel"_L1);
        exptLabel->setGeometry(QRect(9, 9, 53, 16));
        QFont font;
        font.setBold(true);
        // font.setWeight(QFont::Bold);
        exptLabel->setFont(font);
        exptLabel->setAlignment(Qt::AlignCenter);
        splitter = new QSplitter(FtmwViewWidget);
        splitter->setObjectName("splitter"_L1);
        splitter->setGeometry(QRect(9, 27, 821, 471));
        splitter->setOrientation(Qt::Vertical);
        splitter->setChildrenCollapsible(false);
        widget = new QWidget(splitter);
        widget->setObjectName("widget"_L1);
        verticalLayout = new QVBoxLayout(widget);
        verticalLayout->setObjectName("verticalLayout"_L1);
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        livePlotLayout = new QHBoxLayout();
        livePlotLayout->setObjectName("livePlotLayout"_L1);
        liveFidPlot = new FidPlot(QString("Live"),widget);
        liveFidPlot->setObjectName("liveFidPlot"_L1);
//        liveFidPlot->setMinimumSize(QSize(0, 150));

        livePlotLayout->addWidget(liveFidPlot);

        liveFtPlot = new FtPlot(QString("Live"),widget);
        liveFtPlot->setObjectName("liveFtPlot"_L1);

        livePlotLayout->addWidget(liveFtPlot);

        livePlotLayout->setStretch(0, 1);
        livePlotLayout->setStretch(1, 1);

        verticalLayout->addLayout(livePlotLayout,1);

        plots12Layout = new QHBoxLayout();
        plots12Layout->setObjectName("plots12Layout"_L1);
        plot1Layout = new QHBoxLayout();
        plot1Layout->setObjectName("plot1Layout"_L1);
        fidPlot1 = new FidPlot(QString("1"),widget);
        fidPlot1->setObjectName("fidPlot1"_L1);

        plot1Layout->addWidget(fidPlot1);

        ftPlot1 = new FtPlot(QString("1"),widget);
        ftPlot1->setObjectName("ftPlot1"_L1);

        plot1Layout->addWidget(ftPlot1);

        plots12Layout->addLayout(plot1Layout);

        plot2Layout = new QHBoxLayout();
        plot2Layout->setObjectName("plot2ayout"_L1);
        fidPlot2 = new FidPlot(QString("2"),widget);
        fidPlot2->setObjectName("fidPlot2"_L1);

        plot2Layout->addWidget(fidPlot2);

        ftPlot2 = new FtPlot(QString("2"),widget);
        ftPlot2->setObjectName("ftPlot2"_L1);

        plot2Layout->addWidget(ftPlot2);


        plots12Layout->addLayout(plot2Layout);


        verticalLayout->addLayout(plots12Layout,1);

//        verticalLayout->setStretch(0, 1);
//        verticalLayout->setStretch(1, 1);
        splitter->addWidget(widget);
        mainFtPlot = new MainFtPlot(splitter);
        mainFtPlot->setObjectName("mainFtPlot"_L1);
        mainFtPlot->setMinimumSize(QSize(0, 100));
        splitter->addWidget(mainFtPlot);
        splitter->setStretchFactor(0,1);
        splitter->setStretchFactor(1,2);

        toolBar = new QToolBar(FtmwViewWidget);
        toolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        processingAct =toolBar->addAction(QIcon(QString(":/icons/presentation-chart-line.svg")),QString("FID Processing Settings"));
        processingAct->setCheckable(true);

        processingToolBar = new FtmwProcessingToolBar(main,FtmwViewWidget);
        processingToolBar->setVisible(false);


        plotAction = toolBar->addAction(QIcon(QString(":/icons/presentation-chart-bar.svg")),QString("Plot Settings"));
        plotAction->setCheckable(true);

        plotToolBar = new FtmwPlotToolBar(FtmwViewWidget);
        plotToolBar->setVisible(false);

        auto peakupAction = toolBar->addAction(QIcon(":/icons/arrow-trending-up.svg"),QString("Peak Up Options"));
        auto peakupButton = dynamic_cast<QToolButton*>(toolBar->widgetForAction(peakupAction));
        auto peakupMenu = new QMenu;
        auto peakupWa = new QWidgetAction(peakupMenu);
        auto peakupWidget = new QWidget;
        auto peakupFl = new QFormLayout;
        averagesSpinbox = new QSpinBox;
        averagesSpinbox->setRange(1,INT_MAX);
        averagesSpinbox->setEnabled(false);
        averagesSpinbox->setSingleStep(25);
        averagesSpinbox->setKeyboardTracking(false);
        auto avgLbl = new QLabel(QString("Averages"));
        avgLbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
        peakupFl->addRow(avgLbl,averagesSpinbox);

        resetAveragesButton = new QPushButton(QIcon(":/icons/arrow-path.svg"),QString("Reset Averages"));
        resetAveragesButton->setEnabled(false);
        peakupFl->addRow(resetAveragesButton);

        peakupWidget->setLayout(peakupFl);
        peakupWa->setDefaultWidget(peakupWidget);
        peakupMenu->addAction(peakupWa);
        peakupButton->setMenu(peakupMenu);
        peakupButton->setPopupMode(QToolButton::InstantPopup);

        manualBackupAction = toolBar->addAction(QIcon(":/icons/archive-box-arrow-down.svg"),QString("Manual Backup"));
        manualBackupAction->setEnabled(false);
        manualBackupAction->setToolTip(QString("Save a backup snapshot of the current FID list"));
        manualBackupAction->setVisible(main); // hidden entirely in the viewer

        peakFindAction = toolBar->addAction(QIcon(":/icons/magnifying-glass-circle.svg"),QString("Peak Find"));
        peakFindAction->setEnabled(false);

        overlayAction = toolBar->addAction(QIcon(":/icons/squares-plus.svg"),QString("Overlays"));
        overlayAction->setEnabled(false);

        auto *spacer = new QWidget;
        spacer->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);
        toolBar->addWidget(spacer);

        refreshBox = new SpinBoxWidgetAction("Refresh Interval",FtmwViewWidget);
        refreshBox->setRange(500,60000);
        refreshBox->setSingleStep(500);
        refreshBox->setSuffix(" ms");
        toolBar->addAction(refreshBox);

        auto vbl = new QVBoxLayout;
        vbl->addWidget(toolBar,0);
        vbl->addWidget(processingToolBar,0);
        vbl->addWidget(plotToolBar,0);
        vbl->addWidget(exptLabel,0);
        vbl->addWidget(splitter,1);
        FtmwViewWidget->setLayout(vbl);

        retranslateUi(FtmwViewWidget);

        QMetaObject::connectSlotsByName(FtmwViewWidget);
    } // setupUi

    void retranslateUi(QWidget *FtmwViewWidget)
    {
        FtmwViewWidget->setWindowTitle(QApplication::translate("FtmwViewWidget", "Form", 0));
        exptLabel->setText(QApplication::translate("FtmwViewWidget", "Experiment", 0));
    } // retranslateUi

};

namespace Ui {
    class FtmwViewWidget: public Ui_FtmwViewWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // FTMWVIEWWIDGET_H
