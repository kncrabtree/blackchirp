#ifndef LIFDISPLAYWIDGET_H
#define LIFDISPLAYWIDGET_H

#include <QWidget>

#include <QVector>

#include "datastructs.h"
#include "liftrace.h"
#include "lifconfig.h"

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
    void lifShotAcquired(const LifTrace t);
    void prepareForExperiment(const LifConfig c);
    void updatePoint(QPair<QPoint,BlackChirp::LifPoint> val);

    void freqSlice(int delayIndex);
    void delaySlice(int freqIndex);

private:
    Ui::LifDisplayWidget *ui;

    int d_numFrequencyPoints;
    int d_numDelayPoints;
    bool d_delayReverse, d_freqReverse;
    int d_currentSpectrumDelayIndex, d_currentTimeTraceFreqIndex;
    QPair<double,double> d_delayRange;
    QPair<double,double> d_freqRange;
    QVector<BlackChirp::LifPoint> d_lifData;

    void updateSpectrum();
    void updateTimeTrace();

protected:
    void resizeEvent(QResizeEvent *ev);
};

#endif // LIFDISPLAYWIDGET_H
