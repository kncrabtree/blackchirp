#include "mks946.h"
#include <hardware/core/hardwareregistration.h>

using namespace BC::Key::Flow;

// Register hardware implementation using new metaobject system
REGISTER_HARDWARE_META(Mks946, "MKS 946 vacuum transducer controller")
REGISTER_HARDWARE_PROTOCOLS(Mks946, CommunicationProtocol::Rs232)
REGISTER_HARDWARE_SETTINGS(Mks946,
    {BC::Key::Flow::flowChannels, "Flow Channels",
     "Number of mass flow controller channels connected.",
     4, 1, QVariant{}, HwSettingPriority::Important},
    {BC::Key::Flow::pUnits, "Pressure Units",
     "Units for pressure reading display.",
     QString("kTorr"), QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Flow::pMax, "Max Pressure",
     "Full-scale pressure for display scaling.",
     10.0, 0.0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Flow::pDec, "Pressure Decimals",
     "Number of decimal places in pressure display.",
     3, 0, 10, HwSettingPriority::Optional},
    {BC::Key::Flow::address, "Device Address",
     "RS-232 device address (default 253 = broadcast).",
     253, 0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Flow::offset, "Channel Offset",
     "First MFC channel number on the controller (1-indexed).",
     1, 0, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Flow::pressureChannel, "Pressure Channel",
     "Channel number of the pressure transducer input.",
     5, 0, QVariant{}, HwSettingPriority::Optional}
)

Mks946::Mks946(const QString& label, QObject *parent) :
    FlowController(QString(Mks946::staticMetaObject.className()), label, parent)
{
    if(!containsArray(channels))
    {
        std::vector<SettingsMap> l;
        int ch = get(flowChannels,4);
        l.reserve(ch);
        for(int i=0; i<ch; ++i)
            l.push_back({{chUnits,QString("sccm")},{chMax,10000.0},{chDecimals,1}});

        setArray(channels,l,true);
    }

    // Communication defaults
    setDefault(BC::Key::Comm::timeout, 100);
    setDefault(BC::Key::Comm::termChar, QString(";FF"));

    save();
}


bool Mks946::fcTestConnection()
{
    QByteArray resp = mksQuery(QString("MD?"));
    if(!resp.contains(QByteArray("946")))
    {
        d_errorString = QString("Received invalid response to model query. Response: %1").arg(QString(resp));
        return false;
    }

    hwDebug(u"Response: %1"_s.arg(QString(resp)));
    return true;
}

void Mks946::hwSetFlowSetpoint(const int ch, const double val)
{
    if(!isConnected())
    {
        hwError("Cannot set flow setpoint due to a previous communication failure. Reconnect and try again."_L1);
        return;
    }

    bool pidActive = false;

    //first ensure recipe 1 is active
    if(!mksWrite(QString("RCP!1")))
    {
        if(d_errorString.contains(QString("166")))
        {
            pidActive = true;
            if(!mksWrite(QString("PID!OFF")))
            {
                hwError(u"Could not disable PID mode to update channel %1 setpoint. Error: %2"_s.arg(ch+1).arg(d_errorString));
                emit hardwareFailure();
                return;
            }
        }
        else
        {
            hwError(d_errorString);
            emit hardwareFailure();
            return;
        }
    }

    if(!pidActive)
    {
        //make sure ratio recipe 1 is active
        if(!mksWrite(QString("RRCP!1")))
        {
            hwError(d_errorString);
            emit hardwareFailure();
            return;
        }
    }

    if(!mksWrite(QString("RRQ%1!%2").arg(ch+get(offset,1)).arg(val,0,'E',2,QLatin1Char(' '))))
    {
        hwError(d_errorString);
        emit hardwareFailure();
        return;
    }

    if(pidActive)
        hwSetPressureControlMode(true);
}

void Mks946::hwSetPressureSetpoint(const double val)
{
    if(!isConnected())
    {
        hwError("Cannot set pressure setpoint due to a previous communication failure. Reconnect and try again."_L1);
        return;
    }

    bool pidActive = false;

    //first ensure recipe 1 is active
    if(!mksWrite(QString("RCP!1")))
    {
        if(d_errorString.contains(QString("166")))
        {
            pidActive = true;
            if(!mksWrite(QString("PID!OFF")))
            {
                hwError(u"Could not disable PID mode to change pressure setpoint. Error: %1"_s.arg(d_errorString));
                emit hardwareFailure();
                return;
            }
        }
        else
        {
            hwError(d_errorString);
            emit hardwareFailure();
            return;
        }
    }

    if(!mksWrite(QString("RPSP!%1").arg(val*1000.0,1,'E',2,QLatin1Char(' '))))
    {
        hwError(d_errorString);
        emit hardwareFailure();
        return;
    }

    if(pidActive)
        hwSetPressureControlMode(true);
}

double Mks946::hwReadFlowSetpoint(const int ch)
{
    if(!isConnected())
        return 0.0;

    QByteArray resp = mksQuery(QString("RRQ%1?").arg(ch+get(offset,1)));
    bool ok = false;
    double out = resp.mid(2).toDouble(&ok);
    if(!ok)
    {
        hwError(u"Received invalid response to channel %1 setpoint query. Response: %2"_s.arg(ch+1).arg(QString(resp)));
        emit hardwareFailure();
        return -1.0;
    }

    return out;
}

double Mks946::hwReadPressureSetpoint()
{
    if(!isConnected())
        return 0.0;

    QByteArray resp = mksQuery(QString("RPSP?"));
    bool ok = false;
    double out = resp.mid(2).toDouble(&ok);
    if(!ok)
    {
        hwError(u"Received invalid response to pressure setpoint query. Response: %1"_s.arg(QString(resp)));
        emit hardwareFailure();
        return -1.0;
    }

    return out/1000.0; // convert to kTorr
}

double Mks946::hwReadFlow(const int ch)
{
    if(!isConnected())
        return 0.0;

    QByteArray resp = mksQuery(QString("FR%1?").arg(ch+get(offset,1)));
    if(resp.contains(QByteArray("MISCONN")) || resp.contains(QByteArray("NAK")))
        return 0.0;

    bool ok = false;
    double out = resp.toDouble(&ok);
    if(!ok)
    {
        hwError(u"Received invalid response to flow query for channel %1. Response: %2"_s.arg(ch+1).arg(QString(resp)));
        emit hardwareFailure();
        return -1.0;
    }

    return out;
}

double Mks946::hwReadPressure()
{
    if(!isConnected())
        return 0.0;

    QByteArray resp = mksQuery(QString("PR%1?").arg(get(pressureChannel,5)));
    if(resp.contains(QByteArray("LO")))
        return 0.0;

    if(resp.contains(QByteArray("MISCONN")) || resp.contains(QByteArray("NO_GAUGE")))
    {
        hwWarn("No pressure gauge connected."_L1);
        setPressureControlMode(false);
        emit hardwareFailure();
        return -0.0;
    }

    if(resp.contains(QByteArray("ATM")))
        return get(pMax,10.0);

    bool ok = false;
    double out = resp.toDouble(&ok);
    if(ok)
        return out/1000.0; //convert to kTorr

    hwError(u"Could not parse reply to pressure query. Response: %1"_s.arg(QString(resp)));
    emit hardwareFailure();
    return -1.0;
}

void Mks946::hwSetPressureControlMode(bool enabled)
{
    if(!isConnected())
    {
        hwError("Cannot set pressure control mode due to a previous communication failure. Reconnect and try again."_L1);
        return;
    }

    if(enabled)
    {
        QList<QString> chNames;
        chNames << QString("A1") << QString("A2") << QString("B1") << QString("B2") << QString("C1") << QString("C2");
        QString expectedPch = chNames.at(get(pressureChannel,5)-1);

        // If PID is already running, check whether all settings are already correct.
        // If so, leave PID running to avoid disturbing the control loop.
        if(mksQuery(QString("PID?")).contains(QByteArray("ON")))
        {
            bool rcpOk   = mksQuery(QString("RCP?")).trimmed()  == QByteArray("1");
            bool rrcpOk  = mksQuery(QString("RRCP?")).trimmed() == QByteArray("1");
            bool rpchOk  = mksQuery(QString("RPCH?")).contains(expectedPch.toLatin1());
            bool rdchOk  = mksQuery(QString("RDCH?")).contains(QByteArray("Rat"));

            if(rcpOk && rrcpOk && rpchOk && rdchOk)
                return;

            // Settings are wrong; disable PID before reconfiguring
            if(!mksWrite(QString("PID!OFF")))
            {
                hwError(d_errorString);
                emit hardwareFailure();
                return;
            }
        }

        if(!mksWrite(QString("RCP!1")))
        {
            hwError(d_errorString);
            emit hardwareFailure();
            return;
        }

        if(!mksWrite(QString("RRCP!1")))
        {
            hwError(d_errorString);
            emit hardwareFailure();
            return;
        }

        if(!mksWrite(QString("RPCH!%1").arg(expectedPch)))
        {
            hwError(d_errorString);
            emit hardwareFailure();
            return;
        }

        if(!mksWrite(QString("RDCH!Rat")))
        {
            hwError(d_errorString);
            emit hardwareFailure();
            return;
        }

        if(!mksWrite(QString("PID!ON")))
        {
            hwError(d_errorString);
            emit hardwareFailure();
            return;
        }
    }
    else
    {
        if(mksQuery(QString("PID?")).contains(QByteArray("ON")))
        {
            if(!mksWrite(QString("PID!OFF")))
            {
                hwError(d_errorString);
                emit hardwareFailure();
                return;
            }
        }
    }
}

int Mks946::hwReadPressureControlMode()
{
    if(!isConnected())
    {
        return -1;
    }
    QByteArray resp = mksQuery(QString("PID?"));
    if(resp.contains(QByteArray("ON")))
        return 1;
    else if(resp.contains(QByteArray("OFF")))
        return 0;
    else
        hwError(u"Received invalid response to pressure control mode query. Response: %1"_s.arg(QString(resp)));

    return -1;

}

void Mks946::fcInitialize()
{
}

bool Mks946::mksWrite(const QString &cmd)
{
    QByteArray resp = p_comm->queryCmd(QString("@%1%2;FF").arg(get(address,253),3,10,QChar('0')).arg(cmd));
    if(resp.contains(QByteArray("ACK")))
        return true;

    d_errorString = QString("Received invalid response to command %1. Response: %2").arg(cmd).arg(QString(resp));
    return false;
}

QByteArray Mks946::mksQuery(const QString &cmd)
{
    int a = get(address,253);
    QByteArray resp = p_comm->queryCmd(QString("@%1%2;FF").arg(a,3,10,QChar('0')).arg(cmd));

    if(!resp.startsWith(QString("@%1ACK").arg(a,3,10,QChar('0')).toLatin1()))
        return resp;

    //chop off prefix
    return resp.mid(7);
}

void Mks946::sleep(bool b)
{
    if(b)
        setPressureControlMode(false);
}
