#include "trackingplot.h"
#include <qwt6/qwt_legend_label.h>
#include <qwt6/qwt_legend.h>

TrackingPlot::TrackingPlot(QWidget *parent) : QwtPlot(parent)
{
    insertLegend(new QwtLegend(),QwtPlot::BottomLegend);
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

