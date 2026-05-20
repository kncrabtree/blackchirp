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
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)

IntellisysIQPlus::IntellisysIQPlus(const QString& label, QObject *parent) :
    PressureController(QString(IntellisysIQPlus::staticMetaObject.className()), label, false, parent)
{
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
        hwError("Could not read chamber pressure"_L1);
        emit hardwareFailure();
        return nan("");
    }
    int f = resp.indexOf('+');
    int l = resp.size();
    bool ok = false;
    double num = resp.mid(f+1,l-f-1).trimmed().toDouble(&ok);
    if(!ok)
    {
        hwError("Could not parse chamber pressure."_L1);
        hwDebug(u"Could not parse chamber pressure. Response = %1 (Hex: %2)"_s.arg(QString(resp), QString(resp.toHex())));
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
        hwError("Could not read chamber pressure set point"_L1);
        emit hardwareFailure();
        return nan("");
    }

    int f = resp.indexOf('+');
    int l = resp.size();
    bool ok = false;
    double out = resp.mid(f+1,l-f-1).trimmed().toDouble(&ok);
    if(!ok)
    {
        hwError("Could not parse pressure setpoint response."_L1);
        hwDebug(u"Could not parse pressure setpoint response. Response = %1 (Hex: %2)"_s.arg(QString(resp), QString(resp.toHex())));
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
