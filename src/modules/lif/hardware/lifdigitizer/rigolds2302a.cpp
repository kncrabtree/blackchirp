#include "rigolds2302a.h"

#include <QThread>

RigolDS2302A::RigolDS2302A(QObject *parent)
    : LifScope{BC::Key::LifDigi::rds2302a,BC::Key::LifDigi::rds2302aName,
               CommunicationProtocol::Tcp,parent}
{
    using namespace BC::Key::Digi;

    setDefault(numAnalogChannels,2);
    setDefault(numDigitalChannels,0);
    setDefault(hasAuxTriggerChannel,true);
    setDefault(minFullScale,5e-2);
    setDefault(maxFullScale,2.5);
    setDefault(minVOffset,-2.0);
    setDefault(maxVOffset,2.0);
    setDefault(isTriggered,true);
    setDefault(minTrigDelay,-10.0);
    setDefault(maxTrigDelay,10.0);
    setDefault(minTrigLevel,-5.0);
    setDefault(maxTrigLevel,5.0);
    setDefault(maxRecordLength,1400000);
    setDefault(canBlockAverage,false);
    setDefault(canMultiRecord,false);
    setDefault(multiBlock,false);
    setDefault(maxBytes,1);
    setDefault(maxRecords,1);
    setDefault(maxAverages,1);
    setDefault(BC::Key::LifDigi::queryIntervalMs,101);

    if(!containsArray(sampleRates))
        setArray(sampleRates,{
                     {{srText,"10 MSa/s"},{srValue,1e7}},
                     {{srText,"20 MSa/s"},{srValue,2e7}},
                     {{srText,"50 MSa/s"},{srValue,5e7}},
                     {{srText,"100 MSa/s"},{srValue,1e8}},
                     {{srText,"200 MSa/s"},{srValue,2e8}},
                     {{srText,"500 MSa/s"},{srValue,5e8}},
                     {{srText,"1000 MSa/s"},{srValue,1e9}}
                 });

    save();
}


void RigolDS2302A::initialize()
{
    p_comm->setReadOptions(1000,true,"\n");
}

bool RigolDS2302A::testConnection()
{
    auto resp = p_comm->queryCmd("*IDN?\n");

    if(resp.isEmpty())
    {
        //retry once
        QThread::msleep(50);
        resp = p_comm->queryCmd("*IDN?\n");
        if(resp.isEmpty())
        {
            d_errorString = "No response to *IDN query";
            return false;
        }
    }

    if(!resp.startsWith("RIGOL TECHNOLOGIES,DS2302A"))
    {
        d_errorString = QString("Invalid response to *IDN query: %1 (%2)").arg(QString(resp),QString(resp.toHex()));
        return false;
    }

    p_comm->writeCmd("*CLS\n");

    emit logMessage(QString("ID response: %1").arg(QString(resp)));
    return true;
}

void RigolDS2302A::readWaveform()
{
    if(!d_acquiring)
        return;
//    auto resp = p_comm->queryCmd("*OPC?\n");
//    resp = p_comm->queryCmd("*ESR?\n");
//    resp = p_comm->queryCmd("*SRE?\n");
//    resp = p_comm->queryCmd("*STB?\n");
//    if(resp.isEmpty())
//    {
//        d_acquiring = false;
//        emit logMessage("Communication failure.",LogHandler::Error);
//        emit hardwareFailure();
//        return;
//    }

//    if(resp.at(0) != '1')
//        return;

    QVector<qint8> buffer;
    if(d_refEnabled)
        buffer.reserve(d_recordLength*2);
    else
        buffer.reserve(d_recordLength);

    //read channel 1
    int lim = 1;
    if(d_refEnabled)
        lim = 2;

//    p_comm->writeCmd(":STOP\n");
    for(int ch = 0; ch < lim; ch++)
    {
        p_comm->writeCmd(":WAV:SOUR CH1\n");
        p_comm->writeCmd(":WAV:MODE NORM\n");
        p_comm->writeCmd(":WAV:FORM BYTE\n");
        auto resp = p_comm->queryCmd(":WAV:YOR?\n");
        bool ok = false;
        quint8 yor = resp.toInt(&ok);
        if(!ok)
            yor = 0;

        int i=0;
        int cs = 250000;
        do
        {
            int start = i*cs+1;
            int end = i*cs + qMin(d_recordLength-(i*cs),(i+1)*cs);
            p_comm->writeCmd(QString(":WAV:STAR %1\n").arg(start));
            p_comm->writeCmd(QString(":WAV:STOP %1\n").arg(end));
            resp = p_comm->queryCmd(":WAV:DATA?\n");

            if(!resp.startsWith('#'))
            {
                d_acquiring = false;
                emit logMessage(QString("Communication failure parsing WAV:DATA: %1").arg(QString(resp.mid(0,10))),LogHandler::Error);
                emit hardwareFailure();
                return;
            }

            ok = false;
            int dd = resp.mid(1,1).toInt(&ok);
            if(!ok)
            {
                d_acquiring = false;
                emit logMessage(QString("Communication failure parsing WAV:DATA: %1").arg(QString(resp.mid(0,10))),LogHandler::Error);
                emit hardwareFailure();
                return;
            }

            for(int j=2+dd; j<resp.size(); j++)
            {
//                auto b = resp.at(j);

                qint8 dat = static_cast<qint8>(static_cast<qint16>(resp.at(j))-127-yor);
                buffer.append(dat);
            }
            i++;

        } while(i*250000 < d_recordLength);

    }

//    p_comm->writeCmd(":SINGLE\n");
//    p_comm->writeCmd("*OPC\n");
    emit waveformRead(buffer);

}

bool RigolDS2302A::configure(const LifDigitizerConfig &c)
{
    //store all values, then overwrite as needed
    p_comm->writeCmd(QString(":STOP\n"));
    static_cast<LifDigitizerConfig&>(*this) = c;

    d_channelOrder = LifDigitizerConfig::Sequential;
    d_triggerChannel = 0;

    auto it = c.d_analogChannels.find(1);
    if(it == c.d_analogChannels.end() || !it->second.enabled)
    {
        emit logMessage(QString("Channel 1 must be enabled."),LogHandler::Error);
        return false;
    }
    const auto &ch1 = it->second;

    p_comm->writeCmd(QString(":CHAN1:SCALE %1\n").arg(QString::number(ch1.fullScale/5.0,'e',3)));
    p_comm->writeCmd(QString(":CHAN1:OFFSET %1\n").arg(QString::number(ch1.offset,'e',3)));

    it = c.d_analogChannels.find(2);
    if(it != c.d_analogChannels.end())
    {
        if(it->second.enabled)
        {
            const auto &ch2 = it->second;
            p_comm->writeCmd(QString(":CHAN2:SCALE %1\n")
                             .arg(QString::number(ch2.fullScale/5.0,'e',3)));
            p_comm->writeCmd(QString(":CHAN2:OFFSET %1\n")
                             .arg(QString::number(ch2.offset,'e',3)));
        }
    }

    p_comm->writeCmd(":ACQ:TYPE NORM\n");

    //Only certain combinations of sample rate and record length are allowed.
    //The scope uses a "Main Timebase" in units of s/div
    //There are 7 divisions
    //First calculate user's desired record time
    double recTime_s = static_cast<double>(c.d_recordLength)/c.d_sampleRate;
    auto hScale = recTime_s/7.0;

    //"Nice" timebases are 1eN, 2eN, and 5eN
    //choose nearest
    int exponent = static_cast<int>(floor(log10(hScale)));
    auto sc = hScale*pow(10,-exponent);
    double hbase = 1.0;
    if(sc > 1.0)
        hbase = 2.0;
    if(sc > 2.0)
        hbase = 5.0;
    if(sc > 5.0)
    {
        hbase = 1.0;
        exponent++;
    }

    hbase *= pow(10,exponent);
    p_comm->writeCmd(QString(":TIM:SCAL %1\n").arg(QString::number(hbase,'e',3)));


    QThread::msleep(get(BC::Key::LifDigi::queryIntervalMs,101));
    p_comm->writeCmd(":TFORCE\n");
    auto resp = p_comm->queryCmd(":ACQ:SRATE?\n");
    if(resp.isEmpty())
        return false;
    bool ok = false;
    double sr = resp.toDouble(&ok);
    if(!ok || sr < 1e6)
    {
        emit logMessage(QString("Error reading sample rate. Response %1").arg(QString(resp)));
        return false;
    }
    if(abs(c.d_sampleRate-sr)>1e3)
        emit logMessage(QString("Setting sample rate to actual value of %1 Sa/s").arg(QString::number(sr,'e',3)));
    d_sampleRate = sr;

    int recLen = static_cast<int>(round(c.d_sampleRate*hbase*7));
    if(recLen < c.d_recordLength)
    {
        emit logMessage(QString("Setting record length to nearest value: %1").arg(recLen),LogHandler::Warning);
        d_recordLength = recLen;
    }

    //configure trigger
    p_comm->writeCmd(":TRIG:MODE EDGE\n");
    p_comm->writeCmd(":TRIG:EDG:SOUR EXT\n");
    if(c.d_triggerSlope == DigitizerConfig::RisingEdge)
        p_comm->writeCmd(":TRIG:EDG:SLOP POS\n");
    else
        p_comm->writeCmd(":TRIG:EDG:SLOP NEG\n");

    p_comm->writeCmd(QString(":TRIG:EDG:LEV %1\n")
                     .arg(QString::number(c.d_triggerLevel,'f',3)));

    return true;

}

void RigolDS2302A::waitForOpc()
{
    while(true)
    {
        auto resp = p_comm->queryCmd("*OPC?\n",true);
        if(resp.isEmpty() || resp.startsWith('1'))
            break;
        QThread::msleep(10);
    }
}


void RigolDS2302A::beginAcquisition()
{
    p_comm->writeCmd(":RUN\n");
    d_acquiring = true;
    d_timerId = startTimer(get(BC::Key::LifDigi::queryIntervalMs,101));
}

void RigolDS2302A::endAcquisition()
{
    p_comm->writeCmd(":STOP\n");
    d_acquiring = false;
    if(d_timerId >= 0)
        killTimer(d_timerId);
    d_timerId = -1;
}

void RigolDS2302A::timerEvent(QTimerEvent *event)
{
    if(event->timerId() == d_timerId)
    {
        event->accept();
        readWaveform();
    }
}
