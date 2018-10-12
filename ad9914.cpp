#include "ad9914.h"

#include <QSerialPort>
//#include <QTimer>

#include <math.h>

AD9914::AD9914(QObject *parent) : AWG(parent)
{
    d_subKey = QString("ad9914");
    d_prettyName = QString("AD9914 Direct Digital Synthesizer");
    d_commType = CommunicationProtocol::Rs232;
    d_threaded = false;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    d_clockFreqHz = s.value(QString("sampleRate"),3.75e9).toDouble();
    s.setValue(QString("sampleRate"),d_clockFreqHz);
    s.setValue(QString("maxSamples"),1e9);
    s.setValue(QString("minFreq"),0.0);
    s.setValue(QString("maxFreq"),d_clockFreqHz*0.4/1e6);
    s.setValue(QString("hasProtectionPulse"),false);
    s.setValue(QString("hasAmpEnablePulse"),false);
    s.setValue(QString("rampOnly"),true);
    s.endGroup();
    s.endGroup();
}


bool AD9914::testConnection()
{
    if(!p_comm->testConnection())
    {
        emit(connected(false));
        return false;
    }

//    auto port = dynamic_cast<QSerialPort*>(p_comm->device());
//    port->setDataTerminalReady(true);
//    port->setRequestToSend(true);

    p_comm->device()->readAll();
    QByteArray resp = p_comm->queryCmd(QString("ID\n"));
    resp = p_comm->queryCmd(QString("ID\n"));
//    port->setDataTerminalReady(true);

    if(resp.isEmpty())
    {
        emit connected(false,QString("Did not respond to ID query."));
        return false;
    }

    if(!resp.startsWith(QByteArray("AD9914")))
    {
        emit connected(false,QString("ID response invalid. Response: %1").arg(QString(resp)));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp.trimmed())));

    p_comm->writeCmd(QString("IN\n"));

    emit connected();
    return true;
}

void AD9914::initialize()
{
    p_comm->initialize();
    p_comm->setReadOptions(1000,true,QByteArray("\n"));

    testConnection();
//    QTimer::singleShot(2000,this,&AD9914::testConnection);
}

Experiment AD9914::prepareForExperiment(Experiment exp)
{
    if(!exp.ftmwConfig().isEnabled())
    {
        d_enabledForExperiment = false;
        return exp;
    }

    d_enabledForExperiment = true;

    auto rfc = exp.ftmwConfig().rfConfig();
    auto cc = exp.ftmwConfig().chirpConfig();
    auto seg = cc.chirpList().constFirst().constFirst();

    auto clocks = rfc.getClocks();
    if(clocks.contains(BlackChirp::AwgClock))
    {
        d_clockFreqHz = clocks.value(BlackChirp::AwgClock).desiredFreqMHz*1e6;
        QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
        s.beginGroup(d_key);
        s.beginGroup(d_subKey);
        d_clockFreqHz = s.value(QString("sampleRate"),3.75e9).toDouble();
        s.setValue(QString("sampleRate"),d_clockFreqHz);
        s.setValue(QString("maxFreq"),d_clockFreqHz*0.4);
        s.endGroup();
        s.endGroup();

        s.sync();
    }

    //calculate ramp parameters (as close as possible)
    seg.startFreqMHz = 0.0;
    double startFreqHz = 0.0;


    double endFreqHz = seg.endFreqMHz*1e6;


    //to get duration, try smallest Delta t (0001).
    //if step size is too small (< 0x01000000), then increase delta t until appropriate step size is achieved
    int dtVal = 1;
    QByteArray dtHex, stepHex, startHex, endHex;
    bool done = false;

    while(!done)
    {
        double dt = static_cast<double>(dtVal)*24.0/d_clockFreqHz * 1e6; //units microseconds
        int hSteps = static_cast<int>(floor(seg.durationUs/dt));
        double actDur = dt*static_cast<double>(hSteps);

        double rawStep = (endFreqHz-startFreqHz)/static_cast<double>(hSteps-1);
        int stepCode = static_cast<int>(floor(static_cast<double>(Q_INT64_C(4294967296))*rawStep/d_clockFreqHz));
        if(stepCode < 0x01000000)
        {
            dtVal++;
            continue;
        }

        done = true;

        double actStep = stepCode*d_clockFreqHz/static_cast<double>(Q_INT64_C(4294967296));
        double actEndHz = startFreqHz+static_cast<double>(hSteps-1)*actStep;
        seg.durationUs = actDur;
        seg.endFreqMHz = actEndHz/1e6;


        int endFreqCode = static_cast<int>(round(actEndHz / d_clockFreqHz * static_cast<double>(Q_INT64_C(4294967296))));

        startHex = QByteArray("00000000");
        endHex = QString("%1").arg(endFreqCode,8,16,QChar('0')).toLatin1();
        dtHex = QString("%1").arg(dtVal,4,16,QChar('0')).toLatin1();
        stepHex = QString("%1").arg(stepCode,8,16,QChar('0')).toLatin1();
    }

    //store actual chirp settings
    QList<QList<BlackChirp::ChirpSegment>> l;
    QList<BlackChirp::ChirpSegment> l2;
    l2 << seg;
    l << l2;
    cc.setChirpList(l);


    QByteArray resp = p_comm->queryCmd(QString("IN\n"));
    if(!resp.startsWith(QByteArray("SUCCESS")))
    {
        exp.setHardwareFailed();
        exp.setErrorString(QString("Could not initialize %1").arg(d_prettyName));
        emit hardwareFailure();
        return exp;
    }

    resp = p_comm->queryCmd(QString("OE1\n"));
    {
        if(!resp.startsWith(QByteArray("SUCCESS")))
        {
            exp.setHardwareFailed();
            exp.setErrorString(QString("Could not enable %1 DR Over Output").arg(d_prettyName));
            emit hardwareFailure();
            return exp;
        }
    }

    resp = p_comm->queryCmd(QString("LL%1\n").arg(QString(startHex)));
    {
        if(!resp.startsWith(QByteArray("SUCCESS")))
        {
            exp.setHardwareFailed();
            exp.setErrorString(QString("Could not set %1 Lower Limit to %2").arg(d_prettyName).arg(QString(startHex)));
            emit hardwareFailure();
            return exp;
        }
    }

    resp = p_comm->queryCmd(QString("UL%1\n").arg(QString(endHex)));
    {
        if(!resp.startsWith(QByteArray("SUCCESS")))
        {
            exp.setHardwareFailed();
            exp.setErrorString(QString("Could not set %1 Upper Limit to %2").arg(d_prettyName).arg(QString(endHex)));
            emit hardwareFailure();
            return exp;
        }
    }


    //these falling slope parameters should not matter, but set them anyways
    resp = p_comm->queryCmd(QString("FS%1\n").arg(QString(stepHex)));
    {
        if(!resp.startsWith(QByteArray("SUCCESS")))
        {
            exp.setHardwareFailed();
            exp.setErrorString(QString("Could not set %1 Falling Step Size to %2").arg(d_prettyName).arg(QString(stepHex)));
            emit hardwareFailure();
            return exp;
        }
    }
    resp = p_comm->queryCmd(QString("NS%1\n").arg(QString(dtHex)));
    {
        if(!resp.startsWith(QByteArray("SUCCESS")))
        {
            exp.setHardwareFailed();
            exp.setErrorString(QString("Could not set %1 Negative Slope to %2").arg(d_prettyName).arg(QString(dtHex)));
            emit hardwareFailure();
            return exp;
        }
    }


    resp = p_comm->queryCmd(QString("RS%1\n").arg(QString(stepHex)));
    {
        if(!resp.startsWith(QByteArray("SUCCESS")))
        {
            exp.setHardwareFailed();
            exp.setErrorString(QString("Could not set %1 Rising Step Size to %2").arg(d_prettyName).arg(QString(stepHex)));
            emit hardwareFailure();
            return exp;
        }
    }

    resp = p_comm->queryCmd(QString("PS%1\n").arg(QString(dtHex)));
    {
        if(!resp.startsWith(QByteArray("SUCCESS")))
        {
            exp.setHardwareFailed();
            exp.setErrorString(QString("Could not set %1 Positive Slope to %2").arg(d_prettyName).arg(QString(dtHex)));
            emit hardwareFailure();
            return exp;
        }
    }

    rfc.setChirpConfig(cc);
    auto ftmwc = exp.ftmwConfig();
    ftmwc.setRfConfig(rfc);
    exp.setFtmwConfig(ftmwc);

    return exp;

}

void AD9914::beginAcquisition()
{
    if(d_enabledForExperiment)
        p_comm->queryCmd(QString("RD2\n")); //enable ramp, no dwell-high
}

void AD9914::endAcquisition()
{
    if(d_enabledForExperiment)
        p_comm->queryCmd(QString("RD4\n")); //disable ramp
}

void AD9914::readTimeData()
{
}
