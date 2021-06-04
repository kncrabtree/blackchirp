#ifndef MOTORXYPLOT_H
#define MOTORXYPLOT_H

#include <src/modules/motor/gui/motorspectrogramplot.h>

class MotorXYPlot : public MotorSpectrogramPlot
{
    Q_OBJECT
public:
    MotorXYPlot(QWidget *parent = nullptr);
};

#endif // MOTORXYPLOT_H
