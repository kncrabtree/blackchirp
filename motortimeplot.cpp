#include "motortimeplot.h"

#include <QSettings>
#include <qwt6/qwt_plot_picker.h>
#include <qwt6/qwt_picker_machine.h>

MotorTimePlot::MotorTimePlot(QWidget *parent) : ZoomPanPlot(QString("motorTimePlot"),parent)
{
    QSettings s;
    s.beginGroup(d_name);
    p_curve = new QwtPlotCurve;
    p_curve->setRenderHint(QwtPlotItem::RenderAntialiased);
    QColor c = s.value(QString("color"),QPalette().color(QPalette::Text)).value<QColor>();
    p_curve->setPen(QPen(c));
    p_curve->attach(this);
    s.endGroup();

    QwtPlotPicker *picker = new QwtPlotPicker(this->canvas());
    picker->setAxis(QwtPlot::xBottom,QwtPlot::yLeft);
    picker->setStateMachine(new QwtPickerClickPointMachine);
    picker->setMousePattern(QwtEventPattern::MouseSelect1,Qt::RightButton);
    picker->setTrackerMode(QwtPicker::AlwaysOn);
    picker->setTrackerPen(QPen(QPalette().color(QPalette::Text)));
    picker->setEnabled(true);

    QwtText label(QString("P"));
    label.setFont(QFont(QString("sans-serif"),8));
    setAxisTitle(QwtPlot::yLeft,label);

    label.setText(QString::fromUtf16(u"T (µs)"));
    setAxisTitle(QwtPlot::xBottom,label);
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
