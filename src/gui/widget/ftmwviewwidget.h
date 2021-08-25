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
#include <data/storage/settingsstorage.h>
#include <gui/plot/fidplot.h>
#include <gui/plot/ftplot.h>
#include <gui/widget/ftmwprocessingtoolbar.h>
#include <gui/widget/ftmwplottoolbar.h>
#include <gui/widget/toolbarwidgetaction.h>

class QThread;
class FtmwSnapshotWidget;
class PeakFindWidget;

namespace Ui {
class FtmwViewWidget;
}

namespace BC::Key::FtmwView {
static const QString key("FtmwViewWidget");
static const QString refresh("refreshMs");
}

class FtmwViewWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    explicit FtmwViewWidget(QWidget *parent = 0, QString path = QString(""));
    ~FtmwViewWidget();
    void prepareForExperiment(const Experiment &e);

public slots:
    void setLiveUpdateInterval(int intervalms);
    void updateLiveFidList();
    void updateProcessingSettings(FtWorker::FidProcessingSettings s);
    void updatePlotSetting(int id);

    void fidLoadComplete(int id);
    void fidProcessed(const QVector<double> fidData, double spacing, double min, double max, int workerId);
    void ftDone(const Ft ft, int workerId);
    void ftDiffDone(const Ft ft);
    void updateMainPlot();
    void reprocess(const QList<int> ignore = QList<int>());
    void process(int id, const Fid f);
    void processDiff(const Fid f1, const Fid f2);
    void processSideband(RfConfig::Sideband sb);
    void processBothSidebands();

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
    int d_liveTimerId{-1};

    struct WorkerStatus {
        FtWorker *worker;
        bool busy;
        bool reprocessWhenDone;
        QFutureWatcher<void> *p_watcher;
    };

    struct PlotStatus {
        QFutureWatcher<FidList>* p_watcher;
        FidPlot *fidPlot;
        FtPlot *ftPlot;
        Fid fid;
        Ft ft;
        int frame{0}; //only used for plot1 and plot2
        int segment{0}; //only used for plot1 and plot2
        int backup{0}; //only used for plot1 and plot2
        bool loadWhenDone{false};
    };

    QList<int> d_workerIds;
    std::map<int,WorkerStatus> d_workersStatus;
    std::map<int,PlotStatus> d_plotStatus;
    PeakFindWidget *p_pfw{nullptr};
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
    FtmwProcessingToolBar *processingToolBar;
    QAction *plotAction;
    FtmwPlotToolBar *plotToolBar;
    QSpinBox *averagesSpinbox;
    QPushButton *resetAveragesButton;
    QAction *peakFindAction;
    SpinBoxWidgetAction *refreshBox;

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

        processingToolBar = new FtmwProcessingToolBar(FtmwViewWidget);
        processingToolBar->setVisible(false);


        plotAction = toolBar->addAction(QIcon(QString(":/icons/view-media-visualization.svg")),QString("Plot Settings"));
        plotAction->setCheckable(true);

        plotToolBar = new FtmwPlotToolBar(FtmwViewWidget);
        plotToolBar->setVisible(false);

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
