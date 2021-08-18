#ifndef FTMWVIEWWIDGET_H
#define FTMWVIEWWIDGET_H

#include <QWidget>
#include <QtCore/QVariant>
#include <QtWidgets/QAction>
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


#include <data/experiment/experiment.h>
#include <data/analysis/ftworker.h>
#include <gui/plot/fidplot.h>
#include <gui/plot/ftplot.h>
#include <gui/widget/ftmwprocessingtoolbar.h>
#include <gui/widget/ftmwplotconfigwidget.h>

class QThread;
class FtmwSnapshotWidget;
class PeakFindWidget;

namespace Ui {
class FtmwViewWidget;
}

class FtmwViewWidget : public QWidget
{
    Q_OBJECT
public:
    enum MainPlotMode {
        Live,
        FT1,
        FT2,
        FT1mFT2,
        FT2mFT1,
        UpperSB,
        LowerSB,
        BothSB
    };
    Q_ENUM(MainPlotMode)

    explicit FtmwViewWidget(QWidget *parent = 0, QString path = QString(""));
    ~FtmwViewWidget();
    void prepareForExperiment(const Experiment &e);

public slots:
    void updateLiveFidList();
    void updateProcessingSettings(FtWorker::FidProcessingSettings s);
    void changeFrame(int id, int frameNum);
    void changeSegment(int id, int segmentNum);
    void changeBackup(int id, int backupNum);

    void fidLoadComplete(int id);
    void fidProcessed(const QVector<QPointF> fidData, int workerId);
    void ftDone(const Ft ft, int workerId);
    void ftDiffDone(const Ft ft);
    void updateMainPlot();
    void reprocess(const QList<int> ignore = QList<int>());
    void process(int id, const Fid f);
    void processDiff(const Fid f1, const Fid f2);
    void processSideband(RfConfig::Sideband sb);
    void processBothSidebands();
    void updateSidebandFreqs();

    void modeChanged(MainPlotMode newMode);
    void updateBackups();
    void experimentComplete();

    void changeRollingAverageShots(int shots);
    void resetRollingAverage();

    void launchPeakFinder();


private:
    Ui::FtmwViewWidget *ui;

    std::shared_ptr<FidStorageBase> ps_fidStorage;

    FtWorker::FidProcessingSettings d_currentProcessingSettings;
    int d_currentExptNum;
    int d_currentSegment;
    int d_liveTimerId;
    MainPlotMode d_mode;

    struct WorkerStatus {
        FtWorker *worker;
        QThread *thread;
        bool busy;
        bool reprocessWhenDone;
    };

    struct PlotStatus {
        FidPlot *fidPlot;
        FtPlot *ftPlot;
        Fid fid;
        Ft ft;
        int frame{0}; //only used for plot1 and plot2
        int segment{0}; //only used for plot1 and plot2
        int backup{0}; //only used for plot1 and plot2
        std::unique_ptr<QFutureWatcher<FidList>> pu_watcher{std::make_unique<QFutureWatcher<FidList>>()};
        bool loadWhenDone{false};
    };

    QList<int> d_workerIds;
    QHash<int,WorkerStatus> d_workersStatus;
    std::map<int,PlotStatus> d_plotStatus;
    PeakFindWidget *p_pfw;
    QString d_path;
    const int d_liveId = 0, d_mainId = 3, d_plot1Id = 1, d_plot2Id = 2;
    const QString d_shotsString = QString("Shots: %1");

    void updateFid(int id);


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
    FtPlot *mainFtPlot;
    QToolBar *toolBar;
    QAction *processingAct;
    FtmwProcessingToolBar *processingWidget;
    QAction *liveAction;
    QAction *ft1Action;
    QAction *ft2Action;
    QAction *ft12DiffAction;
    QAction *ft21DiffAction;
    QAction *usAction;
    QAction *lsAction;
    QAction *bsAction;
    QSpinBox *mainPlotFollowSpinBox;
    FtmwPlotConfigWidget *plot1ConfigWidget;
    FtmwPlotConfigWidget *plot2ConfigWidget;
    QSpinBox *averagesSpinbox;
    QPushButton *resetAveragesButton;
    QDoubleSpinBox *minFtSegBox;
    QDoubleSpinBox *maxFtSegBox;
    QAction *peakFindAction;

    void setupUi(QWidget *FtmwViewWidget)
    {
        if (FtmwViewWidget->objectName().isEmpty())
            FtmwViewWidget->setObjectName(QStringLiteral("FtmwViewWidget"));
        FtmwViewWidget->resize(850, 520);
        exptLabel = new QLabel(FtmwViewWidget);
        exptLabel->setObjectName(QStringLiteral("exptLabel"));
        exptLabel->setGeometry(QRect(9, 9, 53, 16));
        QFont font;
        font.setBold(true);
        font.setWeight(75);
        exptLabel->setFont(font);
        exptLabel->setAlignment(Qt::AlignCenter);
        splitter = new QSplitter(FtmwViewWidget);
        splitter->setObjectName(QStringLiteral("splitter"));
        splitter->setGeometry(QRect(9, 27, 821, 471));
        splitter->setOrientation(Qt::Vertical);
        splitter->setChildrenCollapsible(false);
        widget = new QWidget(splitter);
        widget->setObjectName(QStringLiteral("widget"));
        verticalLayout = new QVBoxLayout(widget);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        livePlotLayout = new QHBoxLayout();
        livePlotLayout->setObjectName(QStringLiteral("livePlotLayout"));
        liveFidPlot = new FidPlot(QString("Live"),widget);
        liveFidPlot->setObjectName(QStringLiteral("liveFidPlot"));
//        liveFidPlot->setMinimumSize(QSize(0, 150));

        livePlotLayout->addWidget(liveFidPlot);

        liveFtPlot = new FtPlot(QString("Live"),widget);
        liveFtPlot->setObjectName(QStringLiteral("liveFtPlot"));

        livePlotLayout->addWidget(liveFtPlot);

        livePlotLayout->setStretch(0, 1);
        livePlotLayout->setStretch(1, 1);

        verticalLayout->addLayout(livePlotLayout,1);

        plots12Layout = new QHBoxLayout();
        plots12Layout->setObjectName(QStringLiteral("plots12Layout"));
        plot1Layout = new QHBoxLayout();
        plot1Layout->setObjectName(QStringLiteral("plot1Layout"));
        fidPlot1 = new FidPlot(QString("1"),widget);
        fidPlot1->setObjectName(QStringLiteral("fidPlot1"));

        plot1Layout->addWidget(fidPlot1);

        ftPlot1 = new FtPlot(QString("1"),widget);
        ftPlot1->setObjectName(QStringLiteral("ftPlot1"));

        plot1Layout->addWidget(ftPlot1);

        plots12Layout->addLayout(plot1Layout);

        plot2Layout = new QHBoxLayout();
        plot2Layout->setObjectName(QStringLiteral("plot2ayout"));
        fidPlot2 = new FidPlot(QString("2"),widget);
        fidPlot2->setObjectName(QStringLiteral("fidPlot2"));

        plot2Layout->addWidget(fidPlot2);

        ftPlot2 = new FtPlot(QString("2"),widget);
        ftPlot2->setObjectName(QStringLiteral("ftPlot2"));

        plot2Layout->addWidget(ftPlot2);


        plots12Layout->addLayout(plot2Layout);


        verticalLayout->addLayout(plots12Layout,1);

//        verticalLayout->setStretch(0, 1);
//        verticalLayout->setStretch(1, 1);
        splitter->addWidget(widget);
        mainFtPlot = new FtPlot(QString("Main"),splitter);
        mainFtPlot->setObjectName(QStringLiteral("mainFtPlot"));
        mainFtPlot->setMinimumSize(QSize(0, 100));
        splitter->addWidget(mainFtPlot);
        splitter->setStretchFactor(0,1);
        splitter->setStretchFactor(1,2);

        toolBar = new QToolBar(FtmwViewWidget);
        toolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        processingAct =toolBar->addAction(QIcon(QString(":/icons/labplot-xy-fourier-transform-curve.svg")),QString("FID Processing Settings"));
        processingAct->setCheckable(true);

        processingWidget = new FtmwProcessingToolBar(FtmwViewWidget);
        processingWidget->setVisible(false);
        processingWidget->setMovable(true);
        processingWidget->setFloatable(true);


        auto mainModeAct = toolBar->addAction(QIcon(QString(":/icons/view-media-visualization.svg")),QString("Main Plot Mode"));
        auto mmaButton = dynamic_cast<QToolButton*>(toolBar->widgetForAction(mainModeAct));
        auto mmaMenu = new QMenu;
        auto mmaag = new QActionGroup(mmaMenu);
        mmaag->setExclusive(true);

        liveAction = mmaMenu->addAction(QString("Live"));
        liveAction->setCheckable(true);
        mmaag->addAction(liveAction);

        ft1Action = mmaMenu->addAction(QString("FT 1"));
        ft1Action->setCheckable(true);
        mmaag->addAction(ft1Action);

        ft2Action = mmaMenu->addAction(QString("FT 2"));
        ft2Action->setCheckable(true);
        mmaag->addAction(ft2Action);

        ft12DiffAction = mmaMenu->addAction(QString("FT 1 - FT 2"));
        ft12DiffAction->setCheckable(true);
        mmaag->addAction(ft12DiffAction);

        ft21DiffAction = mmaMenu->addAction(QString("FT 2 - FT 1"));
        ft21DiffAction->setCheckable(true);
        mmaag->addAction(ft21DiffAction);

        usAction = mmaMenu->addAction(QString("Upper Sideband"));
        usAction->setCheckable(true);
        mmaag->addAction(usAction);

        lsAction = mmaMenu->addAction(QString("Lower Sideband"));
        lsAction->setCheckable(true);
        mmaag->addAction(lsAction);

        bsAction = mmaMenu->addAction(QString("Both Sidebands"));
        bsAction->setCheckable(true);
        mmaag->addAction(bsAction);

        auto flwWa = new QWidgetAction(mmaMenu);
        auto flwW = new QWidget;
        auto flwFl = new QFormLayout;
        mainPlotFollowSpinBox = new QSpinBox;
        mainPlotFollowSpinBox->setRange(1,2);
        mainPlotFollowSpinBox->setToolTip(QString("When not mirroring another plot or calculating a simple difference, the main plot needs to know what frame, segment, or backup to look at.\n\n(e.g., when plotting the sideband spectra in LO Scan mode). Settings will be taken from the selected plot"));

        auto flwL = new QLabel("Frame/Seg/Backup Follow Plot");
        flwL->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);

        flwFl->addRow(flwL,mainPlotFollowSpinBox);

        minFtSegBox = new QDoubleSpinBox;
        minFtSegBox->setRange(0.0,100.0);
        minFtSegBox->setDecimals(3);
        minFtSegBox->setSuffix(QString(" MHz"));
        minFtSegBox->setKeyboardTracking(false);
        minFtSegBox->setToolTip(QString("Minimum offset frequency included in sideband deconvilution algorithm."));

        flwFl->addRow(QString("Sideband Start"),minFtSegBox);

        maxFtSegBox = new QDoubleSpinBox;
        maxFtSegBox->setRange(0.0,100.0);
        maxFtSegBox->setDecimals(3);
        maxFtSegBox->setValue(100.0);
        maxFtSegBox->setSuffix(QString(" MHz"));
        maxFtSegBox->setKeyboardTracking(false);
        maxFtSegBox->setToolTip(QString("Maximum offset frequency included in sideband deconvilution algorithm."));

        flwFl->addRow(QString("Sideband End"),maxFtSegBox);

        flwW->setLayout(flwFl);
        flwWa->setDefaultWidget(flwW);
        mmaMenu->addAction(flwWa);

        mmaButton->setMenu(mmaMenu);
        mmaButton->setPopupMode(QToolButton::InstantPopup);

        auto plot1Action = toolBar->addAction(QIcon(":/icons/plot1.svg"),QString("Plot 1 Options"));
        auto plot1Button = dynamic_cast<QToolButton*>(toolBar->widgetForAction(plot1Action));
        auto plot1Menu = new QMenu;
        auto plot1wa = new QWidgetAction(plot1Menu);
        plot1ConfigWidget = new FtmwPlotConfigWidget;
        plot1wa->setDefaultWidget(plot1ConfigWidget);
        plot1Menu->addAction(plot1wa);
        plot1Button->setMenu(plot1Menu);
        plot1Button->setPopupMode(QToolButton::InstantPopup);


        auto plot2Action = toolBar->addAction(QIcon(":/icons/plot2.svg"),QString("Plot 2 Options"));
        auto plot2Button = dynamic_cast<QToolButton*>(toolBar->widgetForAction(plot2Action));
        auto plot2Menu = new QMenu;
        auto plot2wa = new QWidgetAction(plot2Menu);
        plot2ConfigWidget = new FtmwPlotConfigWidget;
        plot2wa->setDefaultWidget(plot2ConfigWidget);
        plot2Menu->addAction(plot2wa);
        plot2Button->setMenu(plot2Menu);
        plot2Button->setPopupMode(QToolButton::InstantPopup);

        auto peakupAction = toolBar->addAction(QIcon(":/icons/averaging.svg"),QString("Peak Up Options"));
        auto peakupButton = dynamic_cast<QToolButton*>(toolBar->widgetForAction(peakupAction));
        auto peakupMenu = new QMenu;
        auto peakupWa = new QWidgetAction(peakupMenu);
        auto peakupWidget = new QWidget;
        auto peakupFl = new QFormLayout;
        averagesSpinbox = new QSpinBox;
        averagesSpinbox->setRange(1,__INT_MAX__);
        averagesSpinbox->setEnabled(false);
        averagesSpinbox->setSingleStep(25);
        averagesSpinbox->setKeyboardTracking(false);
        auto avgLbl = new QLabel(QString("Averages"));
        avgLbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
        peakupFl->addRow(avgLbl,averagesSpinbox);

        resetAveragesButton = new QPushButton(QIcon(":/icons/reset.svg"),QString("Reset Averages"));
        resetAveragesButton->setEnabled(false);
        peakupFl->addRow(resetAveragesButton);

        peakupWidget->setLayout(peakupFl);
        peakupWa->setDefaultWidget(peakupWidget);
        peakupMenu->addAction(peakupWa);
        peakupButton->setMenu(peakupMenu);
        peakupButton->setPopupMode(QToolButton::InstantPopup);

        peakFindAction = toolBar->addAction(QIcon(":/icons/peak-find.svg"),QString("Peak Find"));
        peakFindAction->setEnabled(false);

        auto *spacer = new QWidget;
        spacer->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Minimum);
        toolBar->addWidget(spacer);

        auto vbl = new QVBoxLayout;
        vbl->addWidget(toolBar,0);
        vbl->addWidget(processingWidget,0);
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
