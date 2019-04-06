#include "pressurecontroller.h"

PressureController::PressureController(QObject *parent) : HardwareObject(parent)
{
    d_key = QString("pressureController");

    d_readOnly = true;
    d_pressure = 0.0;
    d_setPoint = 0.0;
    d_pressureControlMode = false;
}

PressureController::~PressureController()
{
}


QList<QPair<QString, QVariant> > PressureController::readAuxPlotData()
{
    QList<QPair<QString,QVariant>> out;
    out.append(qMakePair(QString("chamberPressure"),readPressure()));
    return out;
}


void PressureController::initialize()
{
    pcInitialize();
    emit isReadOnly(d_readOnly);
}
