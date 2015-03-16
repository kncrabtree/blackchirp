#include "trackingplot.h"
#include <qwt6/qwt_legend_label.h>
#include <qwt6/qwt_legend.h>

TrackingPlot::TrackingPlot(QWidget *parent) : ZoomPanPlot(parent)
{
    QwtLegend *l = new QwtLegend;
    connect(l,&QwtLegend::checked,this,&TrackingPlot::legendItemClicked);
    insertLegend(l,QwtPlot::BottomLegend);
    setAxisScaleDraw(QwtPlot::xBottom,new TimeScaleDraw);
    setAxisScaleDraw(QwtPlot::xTop,new TimeScaleDraw);

    setAxisFont(QwtPlot::xBottom,QFont(QString("sans-serif"),8));
    setAxisFont(QwtPlot::xTop,QFont(QString("sans-serif"),8));
    setAxisFont(QwtPlot::yLeft,QFont(QString("sans-serif"),8));
    setAxisFont(QwtPlot::yRight,QFont(QString("sans-serif"),8));

    canvas()->installEventFilter(this);
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

void TrackingPlot::legendItemClicked(QVariant info, bool checked, int index)
{
    Q_UNUSED(index);

    QwtPlotCurve *c = dynamic_cast<QwtPlotCurve*>(infoToItem(info));
    if(c != nullptr)
        emit curveVisiblityToggled(c,checked);
}

void TrackingPlot::filterData()
{
}

bool TrackingPlot::eventFilter(QObject *obj, QEvent *ev)
{
    if(obj == legend())
    {
        if(ev->type() == QEvent::MouseButtonRelease)
        {
            QMouseEvent *me = dynamic_cast<QMouseEvent*>(ev);
        }
    }
}

