#ifndef LIFDISPLAYWIDGET_H
#define LIFDISPLAYWIDGET_H

#include <QWidget>

#include <QVector>

#include <data/datastructs.h>
#include <modules/lif/data/liftrace.h>
#include <modules/lif/data/lifconfig.h>

class LifSlicePlot;
class LifTracePlot;
class LifSpectrogramPlot;

namespace BC::Key {
static const QString lifSpectrumPlot("lifSpectrumPlot");
static const QString lifTimePlot("lifTimePlot");
}

class LifDisplayWidget : public QWidget
{
    Q_OBJECT

public:
    explicit LifDisplayWidget(QWidget *parent = 0);
    ~LifDisplayWidget();

signals:
    void lifColorChanged();

public slots:
    void checkLifColors();
    void resetLifPlot();
    void prepareForExperiment(const LifConfig c);
    void updatePoint(const LifConfig c);

    void freqSlice(int delayIndex);
    void delaySlice(int freqIndex);
    void lifZoneUpdate(int min, int max);
    void refZoneUpdate(int min, int max);

private:
    LifConfig d_currentLifConfig;
    bool d_delayReverse, d_freqReverse;
    int d_currentSpectrumDelayIndex, d_currentTimeTraceFreqIndex;
    QVector<double> d_currentIntegratedData;

    LifSlicePlot *p_timeSlicePlot, *p_freqSlicePlot;
    LifTracePlot *p_lifTracePlot;
    LifSpectrogramPlot *p_spectrogramPlot;

    void updateSpectrum();
    void updateTimeTrace();
    void updateLifTrace();

    QPair<double, double> integrate(const LifConfig c);

};

#endif // LIFDISPLAYWIDGET_H
