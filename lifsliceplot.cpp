#include "lifsliceplot.h"

#include <QSettings>

#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_plot_textlabel.h>

LifSlicePlot::LifSlicePlot(QWidget *parent) :
    ZoomPanPlot(QString("lifSlicePlot"),parent)
{
    setAxisFont(QwtPlot::xBottom,QFont(QString("sans-serif"),8));
    setAxisFont(QwtPlot::yLeft,QFont(QString("sans-serif"),8));

    QwtText llabel(QString("LIF (AU)"));
    llabel.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::yLeft,llabel);

    p_curve = new QwtPlotCurve(QString("trace"));
    p_curve->setRenderHint(QwtPlotItem::RenderAntialiased);
    p_curve->setZ(1.0);
    p_curve->attach(this);

}

LifSlicePlot::~LifSlicePlot()
{

}

void LifSlicePlot::setXAxisTitle(QString title)
{
    QwtText label(title);
    label.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::xBottom,label);

    replot();
}

void LifSlicePlot::setName(QString name)
{
    ZoomPanPlot::setName(name);

    QSettings s;
    QColor c = s.value(QString("%1/curveColor").arg(d_name),QPalette().color(QPalette::Text)).value<QColor>();
    p_curve->setPen(QPen(c));
}

void LifSlicePlot::prepareForExperiment(double xMin, double xMax)
{
    p_curve->setSamples(QVector<QPointF>());

    setAxisAutoScaleRange(QwtPlot::xBottom,xMin,xMax);
    setAxisAutoScaleRange(QwtPlot::yLeft,0.0,1.0);

    autoScale();
}

void LifSlicePlot::setData(const QVector<QPointF> d)
{
    p_curve->setSamples(d);
    replot();
}

void LifSlicePlot::setPlotTitle(QString text)
{
    QFont f(QString("sans-serif"),8);
    QwtText title(text);
    title.setFont(f);
    setTitle(title);
}

void LifSlicePlot::filterData()
{
}
