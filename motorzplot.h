#ifndef MOTORZPLOT_H
#define MOTORZPLOT_H

#include "motorspectrogramplot.h"

class MotorZPlot : public MotorSpectrogramPlot
{
    Q_OBJECT
public:
    MotorZPlot(QWidget *parent = nullptr);

};

#endif // MOTORZPLOT_H
