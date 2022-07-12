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

public slots:
    void prepareForExperiment(const Experiment &e);
    void experimentComplete();
    void updatePoint();

    void freqSlice(int delayIndex);
    void delaySlice(int freqIndex);

private:
    std::shared_ptr<LifStorage> ps_lifStorage;
    bool d_delayReverse{false}, d_laserReverse{false};
    QVector<double> d_currentIntegratedData;

    LifSlicePlot *p_timeSlicePlot, *p_freqSlicePlot;
    LifTracePlot *p_lifTracePlot;
    LifSpectrogramPlot *p_spectrogramPlot;
    LifProcessingWidget *p_procWidget;

    void updateSpectrum();
    void updateTimeTrace();
    void updateLifTrace();

    QPair<double, double> integrate(const LifConfig c);

};

#endif // LIFDISPLAYWIDGET_H
