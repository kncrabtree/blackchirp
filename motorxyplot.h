#ifndef MOTORXYPLOT_H
#define MOTORXYPLOT_H

#include "motorspectrogramplot.h"

class MotorXYPlot : public MotorSpectrogramPlot
{
    Q_OBJECT
public:
    MotorXYPlot(QWidget *parent = nullptr);

    // MotorSpectrogramPlot interface
    void buildContextMenu(QMouseEvent *me);
};

#endif // MOTORXYPLOT_H
