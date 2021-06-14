#include <src/hardware/optional/pressurecontroller/pressurecontroller.h>

PressureController::PressureController(const QString subKey, const QString name, CommunicationProtocol::CommType commType, bool readOnly, QObject *parent, bool threaded, bool critical) :
    HardwareObject(BC::Key::pController,subKey,name,commType,parent,threaded,critical), d_readOnly(readOnly)
{
    d_pressure = 0.0;
    d_setPoint = 0.0;
    d_pressureControlMode = false;

    set(BC::Key::pControllerReadOnly,d_readOnly);
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
}
