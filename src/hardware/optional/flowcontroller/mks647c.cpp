#include "mks647c.h"

#include <math.h>

using namespace BC::Key::Flow;

Mks647c::Mks647c(QObject *parent) :
    FlowController(mks647c,mks647cName,CommunicationProtocol::Rs232,parent),
    d_maxTries(5), d_nextRead(0)
{
    double b = 28316.847; //scfm --> sccm conversion
    double c = b/60.0; // scfh --> sccm conversion

    //Gas ranges. see manual for details. All are in SCCM
    d_gasRangeList << 1.0 << 2.0 << 5.0 << 10.0 << 20.0 << 50.0 << 100.0 << 200.0 << 500.0
                   << 1e3 << 2e3 << 5e3 << 1e4 << 2e4 << 5e4 << 1e5 << 2e5 << 4e5 << 5e5 << 1e6
                   << c << 2.0*c << 5.0*c << 1e1*c << 2e1*c << 5e1*c << 1e2*c << 2e2*c << 5e2*c
                   << b << 2.0*b << 5.0*b << 1e1*b << 2e1*b << 5e1*b << 1e2*b << 2e2*b << 5e2*b
                   << 30e3 << 300e3;

    double d = 0.750061683; //bar --> kTorr conversion
    double f = d*1e-5; // Pa --> kTorr conversion

    //pressure ranges. All are in kTorr. Some are redundant (same value, different units in manual. e.g. 3 = 1000 mTorr, 4 = 1 Torr)
    d_pressureRangeList << 1e-6 << 10e-6 << 100e-6 << 1000e-6 << 1e-3 << 10e-3 << 100e-3 << 1000e-3
                        << 1.0 << 1e1 << 1e2 << 1e-6*d << 10e-6*d << 100e-6*d << 1000e-6*d
                        << 1e-3*d << 10e-3*d << 100e-3*d << 1000e-3*d << d << 1e1*d << 1e2*d
                        << f << 1e1*f << 1e2*f << 1e3*f << 1e4*f << 1e5*f << 1e6*f;

    d_pressureRangeIndex = d_pressureRangeList.indexOf(1e1);
    for(int i=0; i<get(flowChannels,4); i++)
    {
        d_rangeIndexList.append(4);
        d_gcfList.append(0.0);
    }

    if(!containsArray(channels))
    {
        std::vector<SettingsMap> l;
        int ch = get(flowChannels,4);
        l.reserve(ch);
        for(int i=0; i<ch; ++i)
            l.push_back({{chUnits,QString("sccm")},{chMax,500.0},{chDecimals,2}});

        setArray(channels,l,true);
    }

    setDefault(pUnits,QString("kTorr"));
    setDefault(pMax,10.0);
    setDefault(pDec,3);
}

bool Mks647c::fcTestConnection()
{

    QByteArray resp = p_comm->queryCmd(QString("ID;\r\n"),true);

    if(resp.isEmpty())
    {
        //try again
        resp = p_comm->queryCmd(QString("ID;\r\n"));
        if(resp.isEmpty())
        {
            d_errorString = QString("Null response to ID query");
            return false;
        }
    }

    if(!resp.startsWith("MGC 647"))
    {
        //try again... stupid firmware bug sometimes causes responses to be repeated twice
        bool done = false;
        while(!done)
        {
            resp.clear();
            resp = p_comm->queryCmd(QString("ID;\r\n"));
            if(resp.isEmpty())
            {
                d_errorString = QString("Null response to ID query");
                return false;
            }
            if(resp.startsWith("MGC 647"))
                done = true;
            else
            {
                d_errorString = QString("Invalid response to ID query (response = %1, hex = %2)")
                               .arg(QString(resp)).arg(QString(resp.toHex()));
                return false;
            }
        }
    }
    emit logMessage(QString("ID response: %1").arg(QString(resp)));
    emit logMessage(QString("Reading all settings. This will take a few seconds..."));
    return true;

}

void Mks647c::fcInitialize()
{
    p_comm->setReadOptions(1000,true,QByteArray("\r\n"));
}

void Mks647c::hwSetFlowSetpoint(const int ch, const double val)
{
    //make sure range and gcf are updated
    readFlow(ch);

    //use full scale range and gcf to convert val to setpoint in units of 0.1% full scale
    int sp = qRound(val/(d_gasRangeList.at(d_rangeIndexList.at(ch))*d_gcfList.at(ch))*1000.0);

    if(sp >=0 && sp <=1100)
        p_comm->writeCmd(QString("FS%1%2;\r\n").arg(ch+1).arg(sp,4,10,QLatin1Char('0')));
    else
        emit logMessage(QString("Flow setpoint (%1) invalid for current range. Converted value = %2 (valid range: 0-1100)").arg(val).arg(sp),LogHandler::Warning);
}

void Mks647c::hwSetPressureSetpoint(const double val)
{
    //make sure we have updated range settings
    readPressure();

    //use full scale range to convert val to setpoint in units of 0.1% full scale
    int sp = (int)round(val/d_pressureRangeList.at(d_pressureRangeIndex)*1000.0);
    if(sp >=0 && sp <=1100)
       p_comm->writeCmd(QString("PS%1;\r\n").arg(sp,4,10,QLatin1Char('0')));

}

double Mks647c::hwReadFlowSetpoint(const int ch)
{
    //although we already know the flow range, we need to query it again because of the stupid firmware bug
    QByteArray resp = mksQueryCmd(QString("RA%1R;\r\n").arg(ch+1),2).trimmed();

    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to flow range query for channel %1.").arg(ch+1),LogHandler::Error);
	   return -1.0;
    }
    //we don't actually care about the range right now...

    //now read the setpoint
    resp = mksQueryCmd(QString("FS%1R;\r\n").arg(ch+1),5).trimmed();

    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to setpoint query for channel %1.").arg(ch+1),LogHandler::Error);
	   return -1.0;
    }

    bool ok = false;
    double sp = resp.toDouble(&ok)/1000.0;
    if(!ok)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read setpoint for channel %1. Response: %2").arg(ch+1).arg(QString(resp)),LogHandler::Error);
	   return -1.0;
    }

    if(resp.toInt() == 0)
        p_comm->writeCmd(QString("OF%1;\r\n").arg(ch+1));
    else
        p_comm->writeCmd(QString("ON%1;\r\n").arg(ch+1));

    double setPoint = sp*d_gasRangeList.at(d_rangeIndexList.at(ch))*d_gcfList.at(ch);

    return setPoint;

}

double Mks647c::hwReadPressureSetpoint()
{
     //although we already know the pressure range, we need to query it again because of the stupid firmware bug
    QByteArray resp = mksQueryCmd(QString("PUR;\r\n"),2).trimmed();

    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to pressure range query."),LogHandler::Error);
	   return -1.0;
    }
    //we don't care about the pressure range....

    //now read pressure serpoint
    resp = mksQueryCmd(QString("PSR;\r\n"),5).trimmed();

    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to pressure query."),LogHandler::Error);
	   return -1.0;
    }

    bool ok = false;
    double d = resp.toDouble(&ok);
    if(!ok)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read pressure. Response: %1").arg(QString(resp)),LogHandler::Error);
	   return -1.0;
    }

    //pressure is in units of 0.1% of full scale
    double setPoint = d*d_pressureRangeList.at(d_pressureRangeIndex)*1e-3;

    return setPoint;
}

double Mks647c::hwReadFlow(const int ch)
{
    //read flow range
    QByteArray resp = mksQueryCmd(QString("RA%1R;\r\n").arg(ch+1),2).trimmed();

    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to flow range query for channel %1.").arg(ch+1),LogHandler::Error);
	   return -1.0;
    }

    bool ok = false;
    int i = resp.toInt(&ok);
    if(!ok || i >= d_gasRangeList.size() || i<0)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read flow range for channel %1. Response: %2")
                        .arg(ch+1).arg(QString(resp)),LogHandler::Error);
	   return -1.0;
    }


    //now read gas correction factor
    resp = mksQueryCmd(QString("GC%1R;\r\n").arg(ch+1),5).trimmed();
    if(resp.length() != 5) //workaround for firmware bug

    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to correction factor query for channel %1.").arg(ch+1),LogHandler::Error);
	   return -1.0;
    }

    ok = false;
    double gcf = resp.toDouble(&ok)/100.0;
    if(!ok)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read correction factor for channel %1. Response: %2")
                        .arg(ch+1).arg(QString(resp)),LogHandler::Error);
	   return -1.0;
    }

    //now read flow
    resp = p_comm->queryCmd(QString("FL%1R;\r\n").arg(ch+1)).trimmed();
    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to flow query for channel %1.").arg(ch+1),LogHandler::Error);
	   return -1.0;
    }

    ok = false;
    double f = resp.toDouble(&ok);
    if(!ok && resp != "-----")
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read flow for channel %1. Response: %2")
                        .arg(ch+1).arg(QString(resp)),LogHandler::Error);
	   return -1.0;
    }

    //convert readings into flow in sccm. f is in 0.1% of full scale, and gcf is fractional
    double flow = -999.9;
    if(ok)
        flow = (f*1e-3*d_gasRangeList.at(i))*gcf;
    else //flow is off scale; return max value
        flow = d_gasRangeList.at(i)*gcf;

    d_rangeIndexList[ch] = i;
    d_gcfList[ch] = gcf;

    return flow;

}

double Mks647c::hwReadPressure()
{
    //read pressure range
    QByteArray resp = mksQueryCmd(QString("PUR;\r\n"),2).trimmed();

    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to pressure range query."),LogHandler::Error);
	   return -1.0;
    }

    bool ok = false;
    int i = resp.trimmed().toInt(&ok);
    if(!ok || i >= d_pressureRangeList.size() || i<0)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read pressure gauge range. Response: %1").arg(QString(resp)),LogHandler::Error);
	   return -1.0;
    }

    //now read pressure
    resp = mksQueryCmd(QString("PR;\r\n"),5).trimmed();

    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to pressure query."),LogHandler::Error);
	   return -1.0;
    }

    ok = false;
    double d = resp.toDouble(&ok);
    if(!ok)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read pressure. Response: %1").arg(QString(resp)),LogHandler::Error);
	   return -1.0;
    }

    //pressure is in units of 0.1% of full scale
    double pressure = d*d_pressureRangeList.at(i)*1e-3;
    d_pressureRangeIndex = i;

    return pressure;
}

void Mks647c::hwSetPressureControlMode(bool enabled)
{
    if(enabled)
        p_comm->writeCmd(QString("PM1;\r\n"));
    else
        p_comm->writeCmd(QString("PM0;\r\n"));

    //now, query state
    readPressureControlMode();
}

int Mks647c::hwReadPressureControlMode()
{
    QByteArray resp = mksQueryCmd(QString("PMR;\r\n"),1).trimmed();

    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to pressure control mode query."),LogHandler::Error);
       return -1;
    }

    bool ok = false;
    int i = resp.toInt(&ok);
    if(!ok)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not parse pressure control mode response. Response: %1").arg(QString(resp)),LogHandler::Error);
       return -1;
    }

    if(i)
        return 1;
    else
        return 0;
}

void Mks647c::poll()
{
    if(d_nextRead < 0 || d_nextRead >= get(flowChannels,4))
    {
        readPressure();
        d_nextRead = 0;
    }
    else
    {
        readFlow(d_nextRead);
        d_nextRead++;
    }
}

void Mks647c::sleep(bool b)
{
    if(b)
    {
       hwSetPressureControlMode(false);
        p_comm->writeCmd(QString("OF0;\r\n"));
    }
    else
        p_comm->writeCmd(QString("ON0;\r\n"));
}

QByteArray Mks647c::mksQueryCmd(QString cmd, int respLength)
{
    QByteArray resp = p_comm->queryCmd(cmd).trimmed();
    if(resp.length() != respLength)
    {
        int tries = 1;
        bool done = false;
        while(!done && tries < d_maxTries)
        {
            tries++;
            resp.clear();
            resp = p_comm->queryCmd(QString("\r\n")).trimmed();
            if(resp.isEmpty() || resp.length() == respLength)
                done = true;
        }
//        emit logMessage(QString("Took %1 tries to get valid response").arg(tries));
    }
    return resp;
}
