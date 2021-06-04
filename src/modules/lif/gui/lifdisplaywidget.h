#ifndef LIFDISPLAYWIDGET_H
#define LIFDISPLAYWIDGET_H

#include <QWidget>

#include <QVector>

#include <src/data/datastructs.h>
#include <src/modules/lif/data/liftrace.h>
#include <src/modules/lif/data/lifconfig.h>

namespace Ui {
class LifDisplayWidget;
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
    Ui::LifDisplayWidget *ui;

    LifConfig d_currentLifConfig;
    bool d_delayReverse, d_freqReverse;
    int d_currentSpectrumDelayIndex, d_currentTimeTraceFreqIndex;
    QVector<double> d_currentIntegratedData;

    void updateSpectrum();
    void updateTimeTrace();
    void updateLifTrace();

    QPair<double, double> integrate(const LifConfig c);
    QString d_laserUnits;


protected:
    void resizeEvent(QResizeEvent *ev);
};

#endif // LIFDISPLAYWIDGET_H
