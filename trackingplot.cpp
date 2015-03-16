#include "trackingplot.h"
#include <qwt6/qwt_legend_label.h>
#include <qwt6/qwt_legend.h>

TrackingPlot::TrackingPlot(QWidget *parent) : ZoomPanPlot(parent)
{
    insertLegend(new QwtLegend(),QwtPlot::BottomLegend);
    setAxisScaleDraw(QwtPlot::xBottom,new TimeScaleDraw);
    setAxisScaleDraw(QwtPlot::xTop,new TimeScaleDraw);

    setAxisFont(QwtPlot::xBottom,QFont(QString("sans-serif"),8));
    setAxisFont(QwtPlot::xTop,QFont(QString("sans-serif"),8));
    setAxisFont(QwtPlot::yLeft,QFont(QString("sans-serif"),8));
    setAxisFont(QwtPlot::yRight,QFont(QString("sans-serif"),8));
}

TrackingPlot::~TrackingPlot()
{

}

void TrackingPlot::initializeLabel(QwtPlotCurve *curve, bool isVisible)
{
    QwtLegendLabel* item = static_cast<QwtLegendLabel*>
            (static_cast<QwtLegend*>(legend())->legendWidget(itemToInfo(curve)));

    item->setItemMode(QwtLegendData::Checkable);
    item->setChecked(isVisible);
}

void TrackingPlot::filterData()
{
}

