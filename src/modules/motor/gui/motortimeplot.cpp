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
    Q_UNUSED(s)
    p_curve->setCurveData(QVector<QPointF>());
}

void MotorTimePlot::updateData(QVector<QPointF> d)
{
    p_curve->setCurveData(d);
    replot();
}
