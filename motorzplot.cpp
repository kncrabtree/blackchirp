#include "motorzplot.h"

#include <QMenu>
#include <QMouseEvent>

MotorZPlot::MotorZPlot(QWidget *parent) : MotorSpectrogramPlot(parent)
{
    setName(QString("motorZPlot"));
    setAxis(QwtPlot::yLeft,MotorScan::MotorY);
    setAxis(QwtPlot::xBottom,MotorScan::MotorZ);
}

void MotorZPlot::buildContextMenu(QMouseEvent *me)
{
    QMenu *m = contextMenu();

    m->popup(me->globalPos());
}
