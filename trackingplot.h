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
    Q_OBJECT

public:
    TrackingPlot(QWidget *parent = 0);
    ~TrackingPlot();

    void initializeLabel(QwtPlotCurve* curve, bool isVisible);

signals:
    void curveVisiblityToggled(QwtPlotCurve*,bool);
    void legendItemRightClicked(QwtPlotCurve*,QMouseEvent*);
    void axisPushRequested();
    void autoScaleAllRequested();

public slots:
    void legendItemClicked(QVariant info, bool checked, int index);
    void buildContextMenu(QMouseEvent *me);


private:

protected:
    void filterData();
    virtual bool eventFilter(QObject *obj, QEvent *ev);

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
