#ifndef TRACKINGPLOT_H
#define TRACKINGPLOT_H

#include "zoompanplot.h"
#include <qwt6/qwt_plot.h>
#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_date_scale_draw.h>
#include <QPair>
#include <QMouseEvent>
#include <QWheelEvent>


class TrackingPlot : public ZoomPanPlot
{
public:
    TrackingPlot(QWidget *parent = 0);
    ~TrackingPlot();

    void initializeLabel(QwtPlotCurve* curve, bool isVisible);

private:

protected:
    void filterData();

};

class TimeScaleDraw : public QwtDateScaleDraw
{
public:
    TimeScaleDraw() : QwtDateScaleDraw() {}
    virtual QwtText label(double v) const
    {
        return QwtDate::toDateTime(v,timeSpec()).toString(QString("M/d\nh:mm"));
    }
};

#endif // TRACKINGPLOT_H
