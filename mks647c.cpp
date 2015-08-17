#include "mks647c.h"

#include "rs232instrument.h"

Mks647c::Mks647c(QObject *parent) :
    FlowController(parent), d_maxTries(5)
{
    d_subKey = QString("mks647c");
    d_prettyName = QString("MKS 647C Flow Control Unit");

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
    for(int i=0; i<BC_FLOW_NUMCHANNELS; i++)
    {
        d_rangeIndexList.append(4);
        d_gcfList.append(0.0);
    }

    p_comm = new Rs232Instrument(d_key,d_subKey,this);
    connect(p_comm,&CommunicationProtocol::logMessage,this,&Mks647c::logMessage);
    connect(p_comm,&CommunicationProtocol::hardwareFailure,this,&Mks647c::hardwareFailure);
}



bool Mks647c::testConnection()
{
    if(!p_comm->testConnection())
    {
        emit connected(false);
        return false;
    }

    p_comm->blockSignals(true);
    QByteArray resp = p_comm->queryCmd(QString("ID;\r\n"));
    p_comm->blockSignals(false);

    if(resp.isEmpty())
    {
        //try again
        resp = p_comm->queryCmd(QString("ID;\r\n"));
        if(resp.isEmpty())
        {
            emit connected(false,QString("%Null response to ID query"));
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
                emit connected(false,QString("Null response to ID query"));
                return false;
            }
            if(resp.startsWith("MGC 647"))
                done = true;
            else
            {
                emit connected(false,QString("Invalid response to ID query (response = %1, hex = %2)")
                               .arg(QString(resp)).arg(QString(resp.toHex())));
                return false;
            }
        }
    }
    emit logMessage(QString("ID response: %1").arg(QString(resp)));
    emit logMessage(QString("Reading all settings. This will take a few seconds..."));

    readAll();

    p_readTimer->start();

    emit connected();
    return true;

}

void Mks647c::initialize()
{
    p_comm->setReadOptions(1000,true,QByteArray("\r\n"));
    p_comm->initialize();
    testConnection();
}

double Mks647c::setFlowSetpoint(const int ch, const double val)
{
    if(ch < 0 || ch >= BC_FLOW_NUMCHANNELS)
        return -1.0;

    //make sure range and gcf are updated
    readFlow(ch);

    //use full scale range and gcf to convert val to setpoint in units of 0.1% full scale
    int sp = qRound(val/(d_gasRangeList.at(d_rangeIndexList.at(ch))*d_gcfList.at(ch)*1000.0));

    if(sp >=0 && sp <=1100)
        p_comm->writeCmd(QString("FS%1%2;\r\n").arg(ch).arg(sp,4,10,QLatin1Char('0')));
    else
        emit logMessage(QString("Flow setpoint (%1) invalid for current range. Converted value = %2 (valid range: 0-1100)").arg(val).arg(sp),BlackChirp::LogWarning);

    return readFlowSetpoint(ch);
}

double Mks647c::setPressureSetpoint(const double val)
{
    //make sure we have updated range settings
    readPressure();

    //use full scale range to convert val to setpoint in units of 0.1% full scale
    int sp = (int)round(val/d_pressureRangeList.at(d_pressureRangeIndex)*1000.0);
    if(sp >=0 && sp <=1100)
       p_comm->writeCmd(QString("PS%1;\r\n").arg(sp,4,10,QLatin1Char('0')));

    return readPressureSetpoint();

}

double Mks647c::readFlowSetpoint(const int ch)
{
    //although we already know the flow range, we need to query it again because of the stupid firmware bug
    QByteArray resp = mksQueryCmd(QString("RA%1R;\r\n").arg(ch+1),2).trimmed();

    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to flow range query for channel %1.").arg(ch+1),BlackChirp::LogError);
	   return -1.0;
    }
    //we don't actually care about the range right now...

    //now read the setpoint
    resp = mksQueryCmd(QString("FS%1R;\r\n").arg(ch+1),5).trimmed();

    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to setpoint query for ch+1annel %1.").arg(ch+1),BlackChirp::LogError);
	   return -1.0;
    }

    bool ok = false;
    double sp = resp.toDouble(&ok)/1000.0;
    if(!ok)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read setpointfor channel %1. Response: %2").arg(ch+1).arg(QString(resp)),BlackChirp::LogError);
	   return -1.0;
    }

    if(resp.toInt() == 0)
        p_comm->writeCmd(QString("OF%1;\r\n").arg(ch+1));
    else
        p_comm->writeCmd(QString("ON%1;\r\n").arg(ch+1));

    double setPoint = sp*d_gasRangeList.at(d_rangeIndexList.at(ch))*d_gcfList.at(ch);
    d_config.set(ch,BlackChirp::FlowSettingSetpoint,setPoint);
    emit flowSetpointUpdate(ch,setPoint);
    return setPoint;

}

double Mks647c::readPressureSetpoint()
{
     //although we already know the pressure range, we need to query it again because of the stupid firmware bug
    QByteArray resp = mksQueryCmd(QString("PUR;\r\n"),2).trimmed();

    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to pressure range query."),BlackChirp::LogError);
	   return -1.0;
    }
    //we don't care about the pressure range....

    //now read pressure serpoint
    resp = mksQueryCmd(QString("PSR;\r\n"),5).trimmed();

    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to pressure query."),BlackChirp::LogError);
	   return -1.0;
    }

    bool ok = false;
    double d = resp.toDouble(&ok);
    if(!ok)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read pressure. Response: %1").arg(QString(resp)),BlackChirp::LogError);
	   return -1.0;
    }

    //pressure is in units of 0.1% of full scale
    double setPoint = d*d_pressureRangeList.at(d_pressureRangeIndex)*1e-3;
    d_config.setPressureSetpoint(setPoint);
    emit pressureSetpointUpdate(setPoint);
    return setPoint;
}

double Mks647c::readFlow(const int ch)
{
    if(ch < 0 || ch >= BC_FLOW_NUMCHANNELS)
        return -1.0;

    //read flow range
    QByteArray resp = mksQueryCmd(QString("RA%1R;\r\n").arg(ch+1),2).trimmed();

    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to flow range query for channel %1.").arg(ch+1),BlackChirp::LogError);
	   return -1.0;
    }

    bool ok = false;
    int i = resp.toInt(&ok);
    if(!ok || i >= d_gasRangeList.size() || i<0)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read flow range for channel %1. Response: %2")
                        .arg(ch+1).arg(QString(resp)),BlackChirp::LogError);
	   return -1.0;
    }


    //now read gas correction factor
    resp = mksQueryCmd(QString("GC%1R;\r\n").arg(ch+1),5).trimmed();
    if(resp.length() != 5) //workaround for firmware bug

    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to correction factor query for channel %1.").arg(ch+1),BlackChirp::LogError);
	   return -1.0;
    }

    ok = false;
    double gcf = resp.toDouble(&ok)/100.0;
    if(!ok)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read correction factor for channel %1. Response: %2")
                        .arg(ch+1).arg(QString(resp)),BlackChirp::LogError);
	   return -1.0;
    }

    //now read flow
    resp = p_comm->queryCmd(QString("FL%1R;\r\n").arg(ch+1)).trimmed();
    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to flow query for channel %1.").arg(ch+1),BlackChirp::LogError);
	   return -1.0;
    }

    ok = false;
    double f = resp.toDouble(&ok);
    if(!ok && resp != "-----")
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read flow for channel %1. Response: %2")
                        .arg(ch+1).arg(QString(resp)),BlackChirp::LogError);
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
    d_config.set(ch,BlackChirp::FlowSettingFlow,flow);
    emit flowUpdate(ch,flow);
    return flow;

}

double Mks647c::readPressure()
{
    //read pressure range
    QByteArray resp = mksQueryCmd(QString("PUR;\r\n"),2).trimmed();

    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to pressure range query."),BlackChirp::LogError);
	   return -1.0;
    }

    bool ok = false;
    int i = resp.trimmed().toInt(&ok);
    if(!ok || i >= d_pressureRangeList.size() || i<0)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read pressure gauge range. Response: %1").arg(QString(resp)),BlackChirp::LogError);
	   return -1.0;
    }

    //now read pressure
    resp = mksQueryCmd(QString("PR;\r\n"),5).trimmed();

    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to pressure query."),BlackChirp::LogError);
	   return -1.0;
    }

    ok = false;
    double d = resp.toDouble(&ok);
    if(!ok)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not read pressure. Response: %1").arg(QString(resp)),BlackChirp::LogError);
	   return -1.0;
    }

    //pressure is in units of 0.1% of full scale
    d_config.setPressure(d*d_pressureRangeList.at(i)*1e-3);
    d_pressureRangeIndex = i;
    emit pressureUpdate(d_config.pressure());

    return d_config.pressure();
}

void Mks647c::setPressureControlMode(bool enabled)
{
    if(enabled)
        p_comm->writeCmd(QString("PM1;\r\n"));
    else
        p_comm->writeCmd(QString("PM0;\r\n"));

    //now, query state
    readPressureControlMode();
}

bool Mks647c::readPressureControlMode()
{
    QByteArray resp = mksQueryCmd(QString("PMR;\r\n"),1).trimmed();

    if(resp.isEmpty())
    {
        emit hardwareFailure();
        emit logMessage(QString("No response to pressure control mode query."),BlackChirp::LogError);
	   return false;
    }

    bool ok = false;
    int i = resp.toInt(&ok);
    if(!ok)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not parse pressure control mode response. Response: %1").arg(QString(resp)),BlackChirp::LogError);
	   return false;
    }

    if(i)
        d_config.setPressureControlMode(true);
    else
        d_config.setPressureControlMode(false);

    emit pressureControlMode(d_config.pressureControlMode());
    return d_config.pressureControlMode();
}

void Mks647c::sleep(bool b)
{
    if(b)
    {
	   setPressureControlMode(false);
        p_comm->writeCmd(QString("OF0;\r\n"));
    }
    else
        p_comm->writeCmd(QString("ON0;\r\n"));

    HardwareObject::sleep(b);
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

Experiment Mks647c::prepareForExperiment(Experiment exp)
{
    return exp;
}

void Mks647c::beginAcquisition()
{

}

void Mks647c::endAcquisition()
{

}
