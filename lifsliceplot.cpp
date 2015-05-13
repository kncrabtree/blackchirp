#include "lifsliceplot.h"

#include <QSettings>

#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_legend.h>

LifSlicePlot::LifSlicePlot(QWidget *parent) :
    ZoomPanPlot(QString("lifSlicePlot"),parent)
{
    setAxisFont(QwtPlot::xBottom,QFont(QString("sans-serif"),8));
    setAxisFont(QwtPlot::yLeft,QFont(QString("sans-serif"),8));

    QwtText llabel(QString("LIF (AU)"));
    llabel.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::yLeft,llabel);

    d_curve = new QwtPlotCurve(QString("trace"));
    d_curve->setRenderHint(QwtPlotItem::RenderAntialiased);
    d_curve->attach(this);

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
    d_curve->setPen(QPen(c));
}



void LifSlicePlot::filterData()
{
}
