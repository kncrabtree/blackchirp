#ifndef TRACKINGPLOT_H
#define TRACKINGPLOT_H

#include <qwt6/qwt_plot.h>
#include <qwt6/qwt_plot_curve.h>


class TrackingPlot : public QwtPlot
{
public:
    TrackingPlot(QWidget *parent = 0);
    ~TrackingPlot();

    void initializeLabel(QwtPlotCurve* curve, bool isVisible);
};

#endif // TRACKINGPLOT_H
