#include "dsox92004a.h"

#include <QTcpSocket>
#include <QTimer>

using namespace BC::Key::FtmwScope;
using namespace BC::Key::Digi;

DSOx92004A::DSOx92004A(QObject *parent) :
    FtmwScope(BC::Key::FtmwScope::dsox92004a,BC::Key::FtmwScope::dsox92004aName,CommunicationProtocol::Tcp,parent)
{
    setDefault(numAnalogChannels,4);
    setDefault(numDigitalChannels,0);
    setDefault(hasAuxTriggerChannel,true);
    setDefault(minFullScale,5e-2);
    setDefault(maxFullScale,2.0);
    setDefault(minVOffset,-2.0);
    setDefault(maxVOffset,2.0);
    setDefault(isTriggered,true);
    setDefault(minTrigDelay,-10.0);
    setDefault(maxTrigDelay,10.0);
    setDefault(minTrigLevel,-5.0);
    setDefault(maxTrigLevel,5.0);
    setDefault(maxRecordLength,100000000);
    setDefault(canBlockAverage,true);
    setDefault(maxAverages,100);
    setDefault(canMultiRecord,true);
    setDefault(maxRecords,100);
    setDefault(multiBlock,false);
    setDefault(maxBytes,2);
    setDefault(bandwidth,20000.0);

    if(!containsArray(sampleRates))
        setArray(sampleRates,{
                     {{srText,"1 GSa/s"},{srValue,1e9}},
                     {{srText,"1.25 GSa/s"},{srValue,1.25e9}},
                     {{srText,"2 GSa/s"},{srValue,2e9}},
                     {{srText,"2.5 GSa/s"},{srValue,2.5e9}},
                     {{srText,"4 GSa/s"},{srValue,4e9}},
                     {{srText,"10 GSa/s"},{srValue,10e9}},
                     {{srText,"20 GSa/s"},{srValue,20e9}},
                     {{srText,"40 GSa/s"},{srValue,40e9}},
                     {{srText,"80 GSa/s"},{srValue,80e9}}
                 });

    save();
}

bool DSOx92004A::testConnection()
{

    QByteArray resp = p_comm->queryCmd(QString("*IDN?\n"));

    if(resp.isEmpty())
    {
        d_errorString = QString("Did not respond to ID query.");
        return false;
    }

    if(resp.length() > 100)
        resp = resp.mid(0,100);

    if(!resp.startsWith(QByteArray("KEYSIGHT TECHNOLOGIES,DSOX92004A")))
    {
        d_errorString = QString("ID response invalid. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex()));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp)));
    return true;

}

void DSOx92004A::initialize()
{
    p_comm->setReadOptions(1000,true,QByteArray("\n"));
    p_socket = dynamic_cast<QTcpSocket*>(p_comm->device());
    p_socket->setSocketOption(QAbstractSocket::LowDelayOption,1);

}

bool DSOx92004A::prepareForExperiment(Experiment &exp)
{
    d_enabledForExperiment = exp.ftmwEnabled();
    if(!d_enabledForExperiment)
        return true;

    auto &config = exp.ftmwConfig()->d_scopeConfig;

    //disable ugly headers
    if(!scopeCommand(QString("*RST;:SYSTEM:HEADER OFF")))
        return false;

    if(!scopeCommand(QString(":DISPLAY:MAIN OFF")))
        return false;

    if(!scopeCommand(QString(":CHANNEL%1:DISPLAY ON").arg(config.d_fidChannel)))
        return false;

    if(!scopeCommand(QString(":CHANNEL%1:INPUT DC50").arg(config.d_fidChannel)))
        return false;

    if(!scopeCommand(QString(":CHANNEL%1:OFFSET 0").arg(config.d_fidChannel)))
        return false;

    if(!scopeCommand(QString(":CHANNEL%1:SCALE %2").arg(config.d_fidChannel)
                     .arg(QString::number(config.d_analogChannels[config.d_fidChannel].fullScale/5.0,'e',3))))
        return false;

    //trigger settings
    QString slope = QString("POS");
    if(config.d_triggerSlope == FallingEdge)
        slope = QString("NEG");
    QString trigCh = QString("AUX");
    if(config.d_triggerLevel > 0)
        trigCh = QString("CHAN%1").arg(config.d_triggerChannel);

    if(!scopeCommand(QString(":TRIGGER:SWEEP TRIGGERED")))
        return false;

    if(!scopeCommand(QString(":TRIGGER:LEVEL %1,%2").arg(trigCh).arg(config.d_triggerLevel,0,'f',3)))
        return false;

    if(!scopeCommand(QString(":TRIGGER:MODE EDGE")))
        return false;

    if(!scopeCommand(QString(":TRIGGER:EDGE:SOURCE %1").arg(trigCh)))
        return false;

    if(!scopeCommand(QString(":TRIGGER:EDGE:SLOPE %1").arg(slope)))
        return false;


    //set trigger position to left of screen
    if(!scopeCommand(QString(":TIMEBASE:REFERENCE LEFT")))
        return false;



    //Data transfer stuff. LSBFirst is faster, and we'll use 2 bytes because averaging
    //will probably be done
    //write data transfer commands
    if(!scopeCommand(QString(":WAVEFORM:SOURCE CHAN%1").arg(config.d_fidChannel)))
        return false;

    if(!scopeCommand(QString(":WAVEFORM:BYTEORDER LSBFIRST")))
        return false;

    if(!scopeCommand(QString(":WAVEFORM:FORMAT WORD")))
        return false;

    if(!scopeCommand(QString(":WAVEFORM:STREAMING ON")))
        return false;

    config.d_byteOrder = BigEndian;
    config.d_bytesPerPoint = 2;


    //now the fast frame/segmented stuff
    if(config.d_multiRecord)
    {
        if(!scopeCommand(QString(":ACQUIRE:MODE SEGMENTED")))
            return false;

        if(!scopeCommand(QString(":ACQUIRE:SEGMENTED:COUNT %1").arg(config.d_numRecords)))
            return false;

        if(!scopeCommand(QString(":WAVEFORM:SEGMENTED:ALL ON")))    
            return false;

    }
    else
    {
        if(!scopeCommand(QString(":ACQUIRE:MODE RTIME")))
            return false;
    }

    //block averaging...
    if(config.d_blockAverage)
    {
        if(!scopeCommand(QString(":ACQUIRE:AVERAGE ON")))
            return false;

        if(!scopeCommand(QString(":ACQUIRE:COUNT %1").arg(config.d_blockAverage)))
            return false;
    }

    //sample rate and point settings
    if(!scopeCommand(QString(":ACQUIRE:SRATE:ANALOG:AUTO OFF")))
        return false;

    if(!scopeCommand(QString(":ACQUIRE:SRATE:ANALOG %1").arg(QString::number(config.d_sampleRate,'g',2))))
        return false;

    if(!scopeCommand(QString(":ACQUIRE:POINTS:AUTO OFF")))
        return false;

    if(!scopeCommand(QString(":ACQUIRE:POINTS:ANALOG %1").arg(config.d_recordLength)))
        return false;


    p_comm->queryCmd(QString("*TRG;*OPC?\n"));

    p_comm->device()->readAll();

    bool done = false;
    while(!done)
    {
        QByteArray resp = p_comm->queryCmd(QString(":SYSTEM:ERROR? STRING\n"));
        if(resp.startsWith('0') || resp.isEmpty())
            break;

        emit logMessage(QString(resp));

    }

    //verify that FID channel was set correctly
    QByteArray resp = p_comm->queryCmd(QString(":WAVEFORM:SOURCE?\n"));
    if(resp.isEmpty() || !resp.contains(QString("CHAN%1").arg(config.d_fidChannel).toLatin1()))
    {
        emit logMessage(QString("Failed to set FID channel. Response to waveform source query: %1 (Hex: %2)")
                        .arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
        return false;
    }

    //read actual offset and vertical scale
    resp = p_comm->queryCmd(QString(":CHAN%1:OFFSET?\n").arg(config.d_fidChannel));
    if(!resp.isEmpty())
    {
        bool ok = false;
        double offset = resp.trimmed().toDouble(&ok);
        if(!ok)
        {
            emit logMessage(QString("Could not parse offset response. Response: %1 (Hex: %2)")
                            .arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
            return false;
        }
        config.d_analogChannels[d_fidChannel].offset = offset;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to offset query."),LogHandler::Error);
        return false;
    }
    resp = p_comm->queryCmd(QString(":CHAN%1:SCALE?\n").arg(config.d_fidChannel));
    if(!resp.isEmpty())
    {
        bool ok = false;
        double scale = resp.trimmed().toDouble(&ok);
        if(!ok)
        {
            emit logMessage(QString("Could not parse scale response. Response: %2 (Hex: %3)")
                            .arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
            return false;
        }
        if(!(fabs(config.d_analogChannels[d_fidChannel].fullScale-scale*5.0) < 0.01))
            emit logMessage(QString("Vertical scale is different than specified. Target: %1 V, Scope setting: %2 V")
                            .arg(QString::number(config.d_analogChannels[d_fidChannel].fullScale,'f',3))
                            .arg(QString::number(scale*5.0,'f',3)),LogHandler::Warning);
        config.d_analogChannels[d_fidChannel].fullScale = scale*5.0;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to scale query."),LogHandler::Error);
        return false;
    }

    //verify sample rate, record length, and horizontal delay
    resp = p_comm->queryCmd(QString(":ACQUIRE:SRATE?\n"));
    if(!resp.isEmpty())
    {
        bool ok = false;
        double sRate = resp.trimmed().toDouble(&ok);
        if(!ok)
        {
            emit logMessage(QString("Sample rate query returned an invalid response. Response: %1 (Hex: %2)")
                            .arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
            return false;
        }
        if(!(fabs(sRate - config.d_sampleRate)<1e6))
        {
            emit logMessage(QString("Could not set sample rate successfully. Target: %1 GS/s, Scope setting: %2 GS/s")
                            .arg(QString::number(config.d_sampleRate/1e9,'f',3))
                            .arg(QString::number(sRate/1e9,'f',3)),LogHandler::Error);
            return false;
        }
        config.d_sampleRate = sRate;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to sample rate query."),LogHandler::Error);
        return false;
    }

    resp = p_comm->queryCmd(QString(":ACQUIRE:POINTS?\n"));
    if(!resp.isEmpty())
    {
        bool ok = false;
        int recLength = resp.trimmed().toInt(&ok);
        if(!ok)
        {
            emit logMessage(QString("Record length query returned an invalid response. Response: %1 (Hex: %2)")
                            .arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
            return false;
        }
        if(!(abs(recLength-config.d_recordLength) < 1000))
        {
            emit logMessage(QString("Record length limited by scope memory. Length will be different than requested. Target: %1, Scope setting: %2")
                            .arg(QString::number(config.d_recordLength))
                            .arg(QString::number(recLength)),LogHandler::Warning);
        }
        config.d_recordLength = recLength;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to record length query."),LogHandler::Error);
        return false;
    }

    resp = p_comm->queryCmd(QString(":TRIGGER:EDGE:SOURCE?\n"));
    if(resp.isEmpty() || !QString(resp).contains(trigCh),Qt::CaseInsensitive)
    {
        emit logMessage(QString("Could not verify trigger channel. Response: %1 (Hex: %2)")
                        .arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
        return false;
    }


    resp = p_comm->queryCmd(QString(":TRIGGER:EDGE:SLOPE?\n"));
    if(resp.isEmpty() || !QString(resp).contains(slope))
    {
        emit logMessage(QString("Could not verify trigger slope. Response: %1 (Hex: %2)")
                        .arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
        return false;
    }

    static_cast<FtmwDigitizerConfig>(*this) = config;
    d_acquiring = false;

    return true;


}

void DSOx92004A::beginAcquisition()
{
    if(d_enabledForExperiment)
    {
        connect(p_socket,&QTcpSocket::readyRead,this,&DSOx92004A::readWaveform);
        p_comm->writeCmd(QString(":SYSTEM:GUI OFF;:DIGITIZE;*OPC?\n"));
//        p_queryTimer->start(100);
    }
}

void DSOx92004A::endAcquisition()
{
    if(d_enabledForExperiment)
    {
        disconnect(p_socket,&QTcpSocket::readyRead,this,&DSOx92004A::readWaveform);
        disconnect(p_socket, &QTcpSocket::readyRead, this, &DSOx92004A::retrieveData);
//        p_queryTimer->stop();
        p_comm->writeCmd(QString("*CLS\n"));
        p_comm->writeCmd(QString(":SYSTEM:GUI ON\n"));
    }
}

void DSOx92004A::readWaveform()
{
    disconnect(p_socket,&QTcpSocket::readyRead,this,&DSOx92004A::readWaveform);
    QByteArray resp = p_socket->readAll();

    if(resp.contains('1'))
    {
        //begin next transfer -- TEST
        p_comm->writeCmd(QString(":DIGITIZE\n"));

        //grab waveform data directly from socket;
//        p_queryTimer->stop();
        p_comm->writeCmd(QString(":WAVEFORM:DATA?\n"));

        connect(p_socket, &QTcpSocket::readyRead, this, &DSOx92004A::retrieveData);
    }


}

void DSOx92004A::retrieveData()
{
    qint64 bytes = d_bytesPerPoint*d_recordLength*d_numRecords;

    if(p_socket->bytesAvailable() < bytes+2)
        return;

    disconnect(p_socket, &QTcpSocket::readyRead, this, &DSOx92004A::retrieveData);

    char c = 0;
    bool gc = false;
    do
    {
        gc = p_socket->getChar(&c);
    } while(c != '#' && gc);

    if(c != '#') //how would this happen?
        return;


    QByteArray out = p_socket->read(bytes);
    emit shotAcquired(out);

    p_socket->readAll();

    connect(p_socket,&QTcpSocket::readyRead,this,&DSOx92004A::readWaveform);
//    p_comm->writeCmd(QString(":DIGITIZE;*OPC?\n"));
    p_comm->writeCmd(QString("*OPC?\n"));
//    p_queryTimer->start(100);

}

bool DSOx92004A::scopeCommand(QString cmd)
{
    QString orig = cmd;
    if(cmd.endsWith(QString("\n")))
        cmd.chop(1);

    cmd.append(QString(";:SYSTEM:ERROR?\n"));
    QByteArray resp = p_comm->queryCmd(cmd,true);
    if(resp.isEmpty())
    {
        emit logMessage(QString("Timed out on query %1").arg(orig),LogHandler::Error);
        return false;
    }

    int val = resp.trimmed().toInt();
    if(val != 0)
    {
        emit logMessage(QString("Received error %1 on query %2").arg(val).arg(orig),LogHandler::Error);
        return false;
    }
    return true;
}
