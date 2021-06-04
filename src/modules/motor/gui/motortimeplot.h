#ifndef MOTORTIMEPLOT_H
#define MOTORTIMEPLOT_H

#include <src/gui/plot/zoompanplot.h>

#include <qwt6/qwt_plot_curve.h>

#include <src/modules/motor/data/motorscan.h>

class MotorTimePlot : public ZoomPanPlot
{
    Q_OBJECT
public:
    MotorTimePlot(QWidget *parent = nullptr);

    void prepareForScan(const MotorScan s);

public slots:
    void updateData(QVector<QPointF> d);

private:
    QwtPlotCurve *p_curve;

    // ZoomPanPlot interface
protected:
    void filterData();
};

#endif // MOTORTIMEPLOT_H
