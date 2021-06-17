#include "motortimeplot.h"

#include <src/gui/plot/blackchirpplotcurve.h>

MotorTimePlot::MotorTimePlot(QWidget *parent) : ZoomPanPlot(BC::Key::motorTimePlot,parent)
{

    p_curve = new BlackchirpPlotCurve(BC::Key::motorTimeCurve);
    p_curve->attach(this);

    setPlotAxisTitle(QwtPlot::yLeft,QString("P"));
    setPlotAxisTitle(QwtPlot::xBottom,QString::fromUtf16(u"Time (µs)"));
}

void MotorTimePlot::prepareForScan(const MotorScan s)
{
    setAxisAutoScaleRange(QwtPlot::xBottom,s.tVal(0),s.tVal(s.tPoints()-1));
    setAxisAutoScaleRange(QwtPlot::yLeft,0.0,1.0);

    p_curve->setSamples(QVector<QPointF>());
}

void MotorTimePlot::updateData(QVector<QPointF> d)
{
    double min = d.constFirst().y();
    double max = d.constFirst().y();
    for(int i=0; i<d.size(); i++)
    {
        min = qMin(d.at(i).y(),min);
        max = qMax(d.at(i).y(),max);
    }

    p_curve->setSamples(d);
    setAxisAutoScaleRange(QwtPlot::yLeft,min,max);
    replot();
}




void MotorTimePlot::filterData()
{
}
