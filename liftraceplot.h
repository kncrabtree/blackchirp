#ifndef LIFTRACEPLOT_H
#define LIFTRACEPLOT_H

#include "zoompanplot.h"

#include "liftrace.h"

class QwtPlotCurve;
class QwtPlotZoneItem;

class LifTracePlot : public ZoomPanPlot
{
    Q_OBJECT
public:
    LifTracePlot(QWidget *parent = nullptr);
    ~LifTracePlot();

signals:
    void colorChanged();

public slots:
    void setNumAverages(int n);
    void newTrace(const LifConfig::LifScopeConfig c, const QByteArray b);
    void traceProcessed(const LifTrace t);

private:
    QwtPlotCurve *p_lif, *p_ref;
    QwtPlotZoneItem *p_lifZone, *p_refZone;
    LifTrace d_currentTrace;
    int d_numAverages;
    bool d_resetNext;
    QPair<int,int> d_lifZoneRange, d_refZoneRange;


    // ZoomPanPlot interface
protected:
    void filterData();
};

#endif // LIFTRACEPLOT_H
