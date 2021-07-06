#ifndef FTMWPLOTCONFIGWIDGET_H
#define FTMWPLOTCONFIGWIDGET_H

#include <QWidget>

#include <data/experiment/experiment.h>

class QThread;
class QListWidget;
class QSpinBox;
class QPushButton;
class QRadioButton;
class SnapWorker;
class QCheckBox;

class FtmwPlotConfigWidget : public QWidget
{
    Q_OBJECT
public:
    explicit FtmwPlotConfigWidget(int id, QString path = QString(""), QWidget *parent = nullptr);
    ~FtmwPlotConfigWidget();

    void prepareForExperiment(const Experiment e);
    void experimentComplete(const Experiment e);
    void snapshotTaken();
    bool isSnapshotActive();

signals:
    void frameChanged(int id, int frameNum);
    void segmentChanged(int id, int segNum);
//    void snapshotsProcessed(int id, const FtmwConfig);
//    void snapshotsFinalized(const FtmwConfig);

public slots:
    void configureSnapControls();
    void process();
//    void processFtmwConfig(const FtmwConfig ref);
//    void processingComplete(const FtmwConfig out);
    void selectAll();
    void selectNone();
    void finalizeSnapshots();
//    void finalizeComplete(const FtmwConfig out);
    void clearAll();

private:
    QSpinBox *p_frameBox, *p_segmentBox;
    QListWidget *p_lw;
    QRadioButton *p_allButton, *p_recentButton, *p_selectedButton;
    QPushButton *p_finalizeButton, *p_selectAllButton, *p_selectNoneButton;
    QCheckBox *p_remainderBox;
    QThread *p_workerThread;
    SnapWorker *p_sw;

    int d_num, d_id;
    bool d_busy, d_updateWhenDone;
    QString d_path;
//    FtmwConfig d_ftmwToProcess;
};

#endif // FTMWPLOTCONFIGWIDGET_H
