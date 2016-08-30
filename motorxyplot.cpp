#include "motorxyplot.h"

#include <QMenu>
#include <QMouseEvent>

MotorXYPlot::MotorXYPlot(QWidget *parent) : MotorSpectrogramPlot(parent)
{
    setName(QString("motorXYPlot"));
    setAxis(QwtPlot::yLeft,MotorScan::MotorY);
    setAxis(QwtPlot::xBottom,MotorScan::MotorX);
}



void MotorXYPlot::buildContextMenu(QMouseEvent *me)
{
    QMenu *m = contextMenu();

    m->popup(me->globalPos());
}
