#ifndef MOTORZPLOT_H
#define MOTORZPLOT_H

#include "motorspectrogramplot.h"

class MotorZPlot : public MotorSpectrogramPlot
{
    Q_OBJECT
public:
    MotorZPlot(QWidget *parent = nullptr);

    // MotorSpectrogramPlot interface
    void buildContextMenu(QMouseEvent *me);
};

#endif // MOTORZPLOT_H
