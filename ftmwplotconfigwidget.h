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

class FtmwPlotConfigWidget : public QWidget
{
    Q_OBJECT
public:
    explicit FtmwPlotConfigWidget(QWidget *parent = nullptr);

    void prepareForExperiment(const Experiment e);

signals:
    void frameChanged(int frameNum);
    void segmentChanged(int segNum);

public slots:

private:
    QSpinBox *p_frameBox, *p_segmentBox;
    QListWidget *p_lw;
    QRadioButton *p_allButton, *p_recentButton, *p_selectedButton;
    QPushButton *p_finalizeButton, *p_selectAllButton, *p_selectNoneButton;
    QThread *p_workerThread;
    SnapWorker *p_sw;

    int d_num;
    bool d_busy, d_updateWhenDone;
    QString d_path;
};

#endif // FTMWPLOTCONFIGWIDGET_H
