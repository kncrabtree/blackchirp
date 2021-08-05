#include "ad9914.h"

#include <QSerialPort>
//#include <QTimer>

#include <math.h>

AD9914::AD9914(QObject *parent) : AWG(BC::Key::AWG::ad9914,BC::Key::AWG::ad9914Name,CommunicationProtocol::Rs232,parent)
{
    setDefault(BC::Key::AWG::rate,3.75e9);
    setDefault(BC::Key::AWG::samples,1e9);
    setDefault(BC::Key::AWG::min,0.0);
    setDefault(BC::Key::AWG::max,3.75e9*0.4/1e6);
    setDefault(BC::Key::AWG::prot,false);
    setDefault(BC::Key::AWG::amp,false);
    setDefault(BC::Key::AWG::rampOnly,true);
    setDefault(BC::Key::AWG::triggered,true);
}


bool AD9914::testConnection()
{
    QByteArray resp;
    int count = 0;

//    dynamic_cast<QSerialPort*>(p_comm->device())->setDataTerminalReady(true);
    while(true)
    {
        count++;
        resp = p_comm->queryCmd(QString("ID\n"));

        if(resp.isEmpty())
        {
            d_errorString = QString("Did not respond to ID query.");
            return false;
        }

        if(!resp.startsWith("SUCCESS"))
            break;

        if(count > 4)
        {
            d_errorString = QString("Could not communicate after 5 attempts.");
            return false;
        }
    }


    if(!resp.startsWith(QByteArray("AD9914")))
    {
       d_errorString = QString("ID response invalid. Response: %1").arg(QString(resp));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp.trimmed())));

    p_comm->writeCmd(QString("IN\n"));

    return true;
}

void AD9914::initialize()
{
    p_comm->setReadOptions(1000,true,QByteArray("\n"));
}

bool AD9914::prepareForExperiment(Experiment &exp)
{
    if(!exp.ftmwEnabled())
    {
        d_enabledForExperiment = false;
        return true;
    }

    d_enabledForExperiment = true;

//    auto rfc = exp.ftmwConfig()->rfConfig();
    auto seg = exp.ftmwConfig()->d_rfConfig.d_chirpConfig.chirpList().constFirst().constFirst();

    auto clocks = exp.ftmwConfig()->d_rfConfig.getClocks();
    if(clocks.contains(RfConfig::AwgRef))
    {
        auto cf = clocks.value(RfConfig::AwgRef).desiredFreqMHz*1e6;
        set(BC::Key::AWG::rate,cf);
        set(BC::Key::AWG::max,cf*0.4);
    }

    auto clockFreqHz = get<double>(BC::Key::AWG::rate);

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
        double dt = static_cast<double>(dtVal)*24.0/clockFreqHz * 1e6; //units microseconds
        int hSteps = static_cast<int>(floor(seg.durationUs/dt));
        double actDur = dt*static_cast<double>(hSteps);

        double rawStep = (endFreqHz-startFreqHz)/static_cast<double>(hSteps-1);
        int stepCode = static_cast<int>(floor(static_cast<double>(Q_INT64_C(4294967296))*rawStep/clockFreqHz));
        if(stepCode < 0x01000000)
        {
            dtVal++;
            continue;
        }

        done = true;

        double actStep = stepCode*clockFreqHz/static_cast<double>(Q_INT64_C(4294967296));
        double actEndHz = startFreqHz+static_cast<double>(hSteps-1)*actStep;
        seg.durationUs = actDur;
        seg.endFreqMHz = actEndHz/1e6;


        int endFreqCode = static_cast<int>(round(actEndHz / clockFreqHz * static_cast<double>(Q_INT64_C(4294967296))));

        startHex = QByteArray("00000000");
        endHex = QString("%1").arg(endFreqCode,8,16,QChar('0')).toLatin1();
        dtHex = QString("%1").arg(dtVal,4,16,QChar('0')).toLatin1();
        stepHex = QString("%1").arg(stepCode,8,16,QChar('0')).toLatin1();
    }

    //store actual chirp settings
    exp.ftmwConfig()->d_rfConfig.setChirpList({{seg}});


    QByteArray resp = p_comm->queryCmd(QString("IN\n"));
    if(!resp.startsWith(QByteArray("SUCCESS")))
    {
        exp.setErrorString(QString("Could not initialize %1").arg(d_name));
        emit hardwareFailure();
        return false;
    }

    d_settingsHex.clear();
    d_settingsHex.reserve(44);
    d_settingsHex.append(startHex);
    d_settingsHex.append(endHex);
    d_settingsHex.append(stepHex);
    d_settingsHex.append(stepHex);
    d_settingsHex.append(dtHex);
    d_settingsHex.append(dtHex);
    d_settingsHex.append("12");


    return true;

}

void AD9914::beginAcquisition()
{
    if(d_enabledForExperiment)
    {
        p_comm->writeCmd(QString("SA%1\n").arg(QString(d_settingsHex)));
//        p_comm->writeCmd(QString("SA%1\n").arg(QString(d_settingsHex)));
    }
}

void AD9914::endAcquisition()
{
    if(d_enabledForExperiment)
        p_comm->writeCmd(QString("IN\n")); //disable ramp
}
