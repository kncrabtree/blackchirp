#ifndef LIFTRACEPLOT_H
#define LIFTRACEPLOT_H

#include "zoompanplot.h"

class LifTracePlot : public ZoomPanPlot
{
    Q_OBJECT
public:
    LifTracePlot(QWidget *parent = nullptr);
    ~LifTracePlot();
};

#endif // LIFTRACEPLOT_H
