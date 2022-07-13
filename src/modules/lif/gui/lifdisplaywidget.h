#ifndef LIFDISPLAYWIDGET_H
#define LIFDISPLAYWIDGET_H

#include <QWidget>

#include <QVector>
#include <memory.h>

#include <data/experiment/experiment.h>

class LifSlicePlot;
class LifTracePlot;
class LifSpectrogramPlot;
class LifProcessingWidget;

namespace BC::Key {
static const QString lifSpectrumPlot{"lifSpectrumPlot"};
static const QString lifTimePlot{"lifTimePlot"};
}

class LifDisplayWidget : public QWidget
{
    Q_OBJECT

public:
    explicit LifDisplayWidget(QWidget *parent = 0);
    ~LifDisplayWidget();

    QVector<QPointF> laserSlice(int delayIndex) const;
    QVector<QPointF> delaySlice(int laserIndex) const;

public slots:
    void prepareForExperiment(const Experiment &e);
    void experimentComplete();
    void updatePoint();

    void changeLaserSlice(int di);
    void changeDelaySlice(int li);
    void reprocess();
    void resetProc();
    void saveProc();

private:
    std::shared_ptr<LifStorage> ps_lifStorage;
    bool d_delayReverse{false}, d_laserReverse{false};
    QVector<double> d_currentIntegratedData;

    LifSlicePlot *p_delaySlicePlot, *p_laserSlicePlot;
    LifTracePlot *p_lifTracePlot;
    LifSpectrogramPlot *p_spectrogramPlot;
    LifProcessingWidget *p_procWidget;

    QString d_dString;
    QString d_lString;
    int d_lDec{2};


};

#endif // LIFDISPLAYWIDGET_H
