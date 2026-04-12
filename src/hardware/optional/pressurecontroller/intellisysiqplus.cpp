#include "intellisysiqplus.h"
#include <hardware/core/hardwareregistration.h>
#include <data/settings/hardwarekeys.h>

using namespace BC::Key::PController;

// Register hardware implementation
REGISTER_HARDWARE_META(IntellisysIQPlus, "Intellisys IQ+ Pressure Controller")
REGISTER_HARDWARE_PROTOCOLS(IntellisysIQPlus, CommunicationProtocol::Rs232)
REGISTER_HARDWARE_SETTINGS(IntellisysIQPlus,
    {BC::Key::PController::min, "Min Pressure",
     "Minimum pressure reading (display range lower bound).",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PController::max, "Max Pressure",
     "Maximum pressure reading (display range upper bound).",
     20.0, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PController::decimals, "Display Decimals",
     "Number of decimal places in pressure display.",
     4, 0, 8, HwSettingPriority::Optional},
    {BC::Key::PController::units, "Pressure Units",
     "Pressure units for display.",
     QString("Torr"), QVariant{}, QVariant{}, HwSettingPriority::Important},
    {BC::Key::PController::readInterval, "Read Interval (ms)",
     "Polling interval for pressure readings in milliseconds.",
     200, 1, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::PController::hasValve, "Has Valve",
     "Device includes a controlled valve output.",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

IntellisysIQPlus::IntellisysIQPlus(const QString& label, QObject *parent) :
    PressureController(QString(IntellisysIQPlus::staticMetaObject.className()), label, false, parent)
{
    // Communication defaults
    setDefault(BC::Key::Comm::timeout, 1000);
    setDefault(BC::Key::Comm::termChar, QString("\r\n"));

    save();
}


bool IntellisysIQPlus::pcTestConnection()
{

    QByteArray resp = p_comm->queryCmd(QString("R38\n"));
    if(resp.isEmpty())
    {
        d_errorString = QString("Pressure controller did not respond");
        return false;
    }
    if(!resp.startsWith("XIQ")) // should be "IQ+3" based on manual, real resp: "XIQ Rev 4.33 27-Feb-2013"
    {
        d_errorString = QString("Wrong pressure controller connected. Model received: %1").arg(QString(resp.trimmed()));
        return false;
    }

    resp = p_comm->queryCmd(QString("R26\n"));
    if(!resp.startsWith("T11"))
    {
        p_comm->writeCmd(QString("T11\n"));
        resp = p_comm->queryCmd(QString("R26\n"));
        if(!resp.startsWith("T11"))
        {
            d_errorString = QString("Could not set the pressure controller to pressure control mode").arg(QString(resp.trimmed()));
            return false;
        }
    }

    resp = p_comm->queryCmd(QString("RN1\n"));
    int f = resp.indexOf('N');
    int l = resp.size();
    d_fullScale = resp.mid(f+2,l-f-2).trimmed().toDouble();

    if(hwReadPressureSetpoint() < 0.0)
        return false;

    //hold the valve at its current position to ensure pressure control is disabled
    setPressureControlMode(false);
    d_pcOn = false;

    return true;
}

void IntellisysIQPlus::pcInitialize()
{
}

double IntellisysIQPlus::hwReadPressure()
{
    QByteArray resp = p_comm->queryCmd(QString("R5\n"));
    if((resp.isEmpty()) || (!resp.startsWith("P+")))
    {
        emit logMessage(QString("Could not read chamber pressure"),LogHandler::Error);
        emit hardwareFailure();
        return nan("");
    }
    int f = resp.indexOf('+');
    int l = resp.size();
    bool ok = false;
    double num = resp.mid(f+1,l-f-1).trimmed().toDouble(&ok);
    if(!ok)
    {
        emit logMessage(QString("Could not parse chamber pressure. Response: %1").arg(QString(resp)),LogHandler::Error);
        emit hardwareFailure();
        return nan("");
    }

    auto p = num * d_fullScale/100.0;
    return p;
}

double IntellisysIQPlus::hwSetPressureSetpoint(const double val)
{
    double num = val * 100.0 / d_fullScale;
    p_comm->writeCmd(QString("S1%1\n").arg(num,0,'f',2,'0'));

    return val;
}

double IntellisysIQPlus::hwReadPressureSetpoint()
{
    QByteArray resp = p_comm->queryCmd(QString("R1\n"));
    if((resp.isEmpty()) || (!resp.startsWith("S1+")))
    {
        emit logMessage(QString("Could not read chamber pressure set point"),LogHandler::Error);
        emit hardwareFailure();
        return nan("");
    }

    int f = resp.indexOf('+');
    int l = resp.size();
    bool ok = false;
    double out = resp.mid(f+1,l-f-1).trimmed().toDouble(&ok);
    if(!ok)
    {
        emit logMessage(QString("Could not parse pressure setpoint response. Received: %1").arg(QString(resp),LogHandler::Error));
        emit hardwareFailure();
        return nan("");
    }

    return out/100.0*d_fullScale;
}

void IntellisysIQPlus::hwSetPressureControlMode(bool enabled)
{
    d_pcOn = enabled;
    if (enabled)
        p_comm->writeCmd(QString("D1\n"));
    else
        p_comm->writeCmd(QString("H\n"));
}

int IntellisysIQPlus::hwReadPressureControlMode()
{
    return d_pcOn;
}

void IntellisysIQPlus::hwOpenGateValve()
{
    p_comm->writeCmd(QString("O\n"));
}

void IntellisysIQPlus::hwCloseGateValve()
{
    p_comm->writeCmd(QString("C\n"));
}
