#include "motorzplot.h"

#include <QMenu>
#include <QMouseEvent>

MotorZPlot::MotorZPlot(QWidget *parent) : MotorSpectrogramPlot(parent)
{
    setName(QString("motorZPlot"));
    setAxis(QwtPlot::yLeft,MotorScan::MotorY);
    setAxis(QwtPlot::xBottom,MotorScan::MotorZ);

    p_yzData = nullptr;
    p_xzData = nullptr;
}

void MotorZPlot::buildContextMenu(QMouseEvent *me)
{
    QMenu *m = contextMenu();

    m->popup(me->globalPos());
}

void MotorZPlot::prepareForScan(const MotorScan s)
{
    if(p_xzData != nullptr)
    {
        delete p_xzData;
        p_xzData = nullptr;
    }
    if(p_yzData != nullptr)
    {
        delete p_yzData;
        p_yzData = nullptr;
    }

    MotorSpectrogramPlot::prepareForScan(s);

    QwtMatrixRasterData *otherData;
    otherData = new QwtMatrixRasterData;

    if(d_leftAxis == MotorScan::MotorY)
    {
        p_yzData = p_spectrogramData;
        p_xzData = otherData;
    }
    else
    {
        p_xzData = p_spectrogramData;
        p_yzData = otherData;
    }

}


void MotorZPlot::updateData(QVector<double> data, int cols, double max, MotorScan::MotorDataAxis leftAxis, MotorScan::MotorDataAxis bottomAxis)
{
    MotorSpectrogramPlot::updateData(data,cols,max,leftAxis,bottomAxis);

    if(d_leftAxis != leftAxis)
    {
        if(leftAxis == MotorScan::MotorY)
            p_yzData->setValueMatrix(data,cols);
        else if(leftAxis == MotorScan::MotorX)
            p_xzData->setValueMatrix(data,cols);
    }
}
