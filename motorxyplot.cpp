#include "motorxyplot.h"

#include <QMenu>
#include <QMouseEvent>

MotorXYPlot::MotorXYPlot(QWidget *parent) : MotorSpectrogramPlot(parent)
{
    setName(QString("motorXYPlot"));
    setAxis(QwtPlot::yLeft,BlackChirp::MotorY);
    setAxis(QwtPlot::xBottom,BlackChirp::MotorX);
}



void MotorXYPlot::buildContextMenu(QMouseEvent *me)
{
    QMenu *m = contextMenu();

    m->popup(me->globalPos());
}
