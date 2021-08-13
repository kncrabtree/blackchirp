#ifndef FTMWPLOTCONFIGWIDGET_H
#define FTMWPLOTCONFIGWIDGET_H

#include <QWidget>

#include <data/experiment/experiment.h>

class QSpinBox;

class FtmwPlotConfigWidget : public QWidget
{
    Q_OBJECT
public:
    explicit FtmwPlotConfigWidget(QWidget *parent = nullptr);
    ~FtmwPlotConfigWidget();

    void prepareForExperiment(const Experiment &e);
    void newBackup(int numBackups);
    bool viewingBackup();

signals:
    void frameChanged(int);
    void segmentChanged(int);
    void backupChanged(int);

private:
    QSpinBox *p_frameBox, *p_segmentBox, *p_backupBox;

    int d_id;
};

#endif // FTMWPLOTCONFIGWIDGET_H
