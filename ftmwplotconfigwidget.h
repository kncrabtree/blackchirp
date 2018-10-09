#ifndef FTMWPLOTCONFIGWIDGET_H
#define FTMWPLOTCONFIGWIDGET_H

#include <QWidget>

#include "experiment.h"

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
    explicit FtmwPlotConfigWidget(QString path = QString(""), QWidget *parent = nullptr);
    ~FtmwPlotConfigWidget();

    void prepareForExperiment(const Experiment e);
    void experimentComplete(const Experiment e);
    void snapshotTaken();

signals:
    void frameChanged(int frameNum);
    void segmentChanged(int segNum);

public slots:
    void configureSnapControls();

private:
    QSpinBox *p_frameBox, *p_segmentBox;
    QListWidget *p_lw;
    QRadioButton *p_allButton, *p_recentButton, *p_selectedButton;
    QPushButton *p_finalizeButton, *p_selectAllButton, *p_selectNoneButton;
    QCheckBox *p_remainderBox;
    QThread *p_workerThread;
    SnapWorker *p_sw;

    int d_num;
    bool d_busy, d_updateWhenDone;
    QString d_path;
};

#endif // FTMWPLOTCONFIGWIDGET_H
