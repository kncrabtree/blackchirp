#ifndef TRACKINGPLOT_H
#define TRACKINGPLOT_H

#include <qwt6/qwt_text.h>

#include <gui/plot/zoompanplot.h>

class QwtPlotCurve;


class TrackingPlot : public ZoomPanPlot
{
    Q_OBJECT

public:
    TrackingPlot(QString name, QWidget *parent = 0);
    ~TrackingPlot();


signals:
    void axisPushRequested();
    void autoScaleAllRequested();

public slots:
    void buildContextMenu(QMouseEvent *me);

};

#include <qwt6/qwt_date_scale_draw.h>

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
