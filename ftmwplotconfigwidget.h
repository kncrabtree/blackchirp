#ifndef FTMWPLOTCONFIGWIDGET_H
#define FTMWPLOTCONFIGWIDGET_H

#include <QWidget>

#include "experiment.h"

class QSpinBox;

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
};

#endif // FTMWPLOTCONFIGWIDGET_H
