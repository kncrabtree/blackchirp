#ifndef FTMWPLOTCONFIGWIDGET_H
#define FTMWPLOTCONFIGWIDGET_H

#include <QWidget>

#include <data/experiment/experiment.h>

class QSpinBox;

class FtmwPlotConfigWidget : public QWidget
{
    Q_OBJECT
public:
    explicit FtmwPlotConfigWidget(int id, QWidget *parent = nullptr);
    ~FtmwPlotConfigWidget();

    void prepareForExperiment(const Experiment &e);
    void newAutosave(int numAutosaves);
    bool viewingAutosave();

signals:
    void frameChanged(int id, int frameNum);
    void segmentChanged(int id, int segNum);

private:
    QSpinBox *p_frameBox, *p_segmentBox, *p_autosaveBox;

    int d_id;
};

#endif // FTMWPLOTCONFIGWIDGET_H
