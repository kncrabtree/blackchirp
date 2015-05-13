#ifndef LIFSLICEPLOT_H
#define LIFSLICEPLOT_H

#include "zoompanplot.h"

class QwtPlotCurve;

class LifSlicePlot : public ZoomPanPlot
{
    Q_OBJECT
public:
    LifSlicePlot(QWidget *parent = nullptr);
    ~LifSlicePlot();

    void setXAxisTitle(QString title);
    void setName(QString name);

    void prepareForExperiment(double xMin, double xMax);

    // ZoomPanPlot interface
protected:
    void filterData();


    QwtPlotCurve *d_curve;
};

#endif // LIFSLICEPLOT_H
