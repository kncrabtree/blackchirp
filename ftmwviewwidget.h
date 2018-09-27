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
#include <QList>


#include "experiment.h"
#include "ftworker.h"
#include "fidplot.h"
#include "ftplot.h"
#include "ftmwprocessingwidget.h"

class QThread;
class FtmwSnapshotWidget;

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

    explicit FtmwViewWidget(QWidget *parent = 0, QString path = QString(""));
    ~FtmwViewWidget();
    void prepareForExperiment(const Experiment e);

signals:
    void rollingAverageShotsChanged(int);
    void rollingAverageReset();
    void experimentLogMessage(int,QString,BlackChirp::LogMessageCode = BlackChirp::LogNormal,QString=QString(""));
    void finalized(int);

public slots:
    void updateLiveFidList(const FidList fl, int segment);
    void updateFtmw(const FtmwConfig f);
    void updateProcessingSettings(FtWorker::FidProcessingSettings s);
    void storeProcessingSettings();

    void fidProcessed(const QVector<QPointF> fidData, int workerId);
    void ftDone(const Ft ft, int workerId);
    void updateMainPlot();
    void reprocessAll();
    void reprocess(const QList<int> ignore = QList<int>());
    void process(int id, const Fid f);

    void modeChanged(MainPlotMode newMode);
    void snapshotTaken();
    void experimentComplete(const Experiment e);
    void snapshotLoadError(QString msg);
    void snapListUpdate();
    void snapRefChanged();
    void finalizedSnapList(const FidList l);


private:
    Ui::FtmwViewWidget *ui;

    FtmwConfig d_ftmwConfig;
    FtWorker::FidProcessingSettings d_currentProcessingSettings;
    int d_currentExptNum;
    MainPlotMode d_mode;
    FidList d_liveFidList;

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
        int frame; //only used for plot1 and plot2
        int segment; //only used for plot1 and plot2
    };

    QList<int> d_workerIds;
    QHash<int,WorkerStatus> d_workersStatus;
    QHash<int,PlotStatus> d_plotStatus;
    QString d_path;
    const int d_liveFtwId = 0, d_mainFtwId = 3, d_plot1FtwId = 1, d_plot2FtwId = 2;

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
    QVBoxLayout *fid1Layout;
    QHBoxLayout *plot1Layout;
    FidPlot *fidPlot1;
    FtPlot *ftPlot1;
    QWidget *snapshotWidget1;
    QVBoxLayout *fid2Layout;
    QHBoxLayout *plot2ayout;
    FidPlot *fidPlot2;
    FtPlot *ftPlot2;
    QWidget *snapshotWidget2;
    FtPlot *mainFtPlot;
    QToolBar *toolBar;
    QMenu *processingMenu;
    FtmwProcessingWidget *processingWidget;

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
        fid1Layout = new QVBoxLayout();
        fid1Layout->setObjectName(QStringLiteral("fid1Layout"));
        plot1Layout = new QHBoxLayout();
        plot1Layout->setObjectName(QStringLiteral("plot1Layout"));
        fidPlot1 = new FidPlot(QString("1"),widget);
        fidPlot1->setObjectName(QStringLiteral("fidPlot1"));

        plot1Layout->addWidget(fidPlot1);

        ftPlot1 = new FtPlot(QString("1"),widget);
        ftPlot1->setObjectName(QStringLiteral("ftPlot1"));

        plot1Layout->addWidget(ftPlot1);


        fid1Layout->addLayout(plot1Layout);

        snapshotWidget1 = new QWidget(widget);
        snapshotWidget1->setObjectName(QStringLiteral("snapshotWidget1"));

        fid1Layout->addWidget(snapshotWidget1);

        fid1Layout->setStretch(0, 1);
        fid1Layout->setStretch(1, 1);

        plots12Layout->addLayout(fid1Layout);

        fid2Layout = new QVBoxLayout();
        fid2Layout->setObjectName(QStringLiteral("fid2Layout"));
        plot2ayout = new QHBoxLayout();
        plot2ayout->setObjectName(QStringLiteral("plot2ayout"));
        fidPlot2 = new FidPlot(QString("2"),widget);
        fidPlot2->setObjectName(QStringLiteral("fidPlot2"));

        plot2ayout->addWidget(fidPlot2);

        ftPlot2 = new FtPlot(QString("2"),widget);
        ftPlot2->setObjectName(QStringLiteral("ftPlot2"));

        plot2ayout->addWidget(ftPlot2);


        fid2Layout->addLayout(plot2ayout);

        snapshotWidget2 = new QWidget(widget);
        snapshotWidget2->setObjectName(QStringLiteral("snapshotWidget2"));

        fid2Layout->addWidget(snapshotWidget2);

        fid2Layout->setStretch(0, 1);
        fid2Layout->setStretch(1, 1);

        plots12Layout->addLayout(fid2Layout);


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

        toolBar = new QToolBar;
        auto *processingAct =toolBar->addAction(QIcon(QString(":/icons/labplot-xy-fourier-transform-curve.svg")),QString("FID Processing Settings"));
        auto *processingBtn = dynamic_cast<QToolButton*>(toolBar->widgetForAction(processingAct));
        processingMenu = new QMenu;
        auto processingWa = new QWidgetAction(processingMenu);
        processingWidget = new FtmwProcessingWidget;
        processingWa->setDefaultWidget(processingWidget);
        processingMenu->addAction(processingWa);
        processingBtn->setMenu(processingMenu);
        processingBtn->setPopupMode(QToolButton::InstantPopup);



        auto vbl = new QVBoxLayout;
        vbl->addWidget(toolBar,0);
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
