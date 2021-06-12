#include "dsox92004a.h"

#include <QTcpSocket>
#include <QTimer>

DSOx92004A::DSOx92004A(QObject *parent) :
    FtmwScope(BC::Key::dsox92004a,BC::Key::dsox92004aName,CommunicationProtocol::Tcp,parent)
{
}

void DSOx92004A::readSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    s.setValue(QString("canBlockAverage"),true);
    s.setValue(QString("canFastFrame"),true);
    s.setValue(QString("canSummaryFrame"),false);
    s.setValue(QString("canBlockAndFastFrame"),false);

    double bandwidth = s.value(QString("bandwidth"),20000.0).toDouble();
    s.setValue(QString("bandwidth"),bandwidth);

    if(s.beginReadArray(QString("sampleRates")) < 1)
    {
        s.endArray();
        QList<QPair<QString,double>> sampleRates;
        sampleRates << qMakePair(QString("1 GSa/s"),1e9) << qMakePair(QString("1.25 GSa/s"),1.25e9)  << qMakePair(QString("2 GSa/s"),2e9)
                    << qMakePair(QString("2.5 GSa/s"),2.5e9) << qMakePair(QString("4 GSa/s"),4e9) << qMakePair(QString("5 GSa/s"),5e9)  << qMakePair(QString("10 GSa/s"),10e9)
                    << qMakePair(QString("20 GSa/s"),20e9) << qMakePair(QString("40 GSa/s"),40e9)  << qMakePair(QString("80 GSa/s"),80e9);

        s.beginWriteArray(QString("sampleRates"));
        for(int i=0; i<sampleRates.size(); i++)
        {
            s.setArrayIndex(i);
            s.setValue(QString("text"),sampleRates.at(i).first);
            s.setValue(QString("val"),sampleRates.at(i).second);
        }
        s.endArray();
    }
    s.endGroup();
    s.endGroup();
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
    d_enabledForExperiment = exp.ftmwConfig().isEnabled();
    if(!d_enabledForExperiment)
        return true;

    auto config = exp.ftmwConfig().scopeConfig();

    //disable ugly headers
    if(!scopeCommand(QString("*RST;:SYSTEM:HEADER OFF")))
    {
        exp.setHardwareFailed();
        return false;
    }

    if(!scopeCommand(QString(":DISPLAY:MAIN OFF")))
    {
        exp.setHardwareFailed();
        return false;
    }

    if(!scopeCommand(QString(":CHANNEL%1:DISPLAY ON").arg(config.fidChannel)))
    {
        exp.setHardwareFailed();
        return false;
    }
    if(!scopeCommand(QString(":CHANNEL%1:INPUT DC50").arg(config.fidChannel)))
    {
        exp.setHardwareFailed();
        return false;
    }
    if(!scopeCommand(QString(":CHANNEL%1:OFFSET 0").arg(config.fidChannel)))
    {
        exp.setHardwareFailed();
        return false;
    }
    if(!scopeCommand(QString(":CHANNEL%1:SCALE %2").arg(config.fidChannel).arg(QString::number(config.vScale,'e',3))))
    {
        exp.setHardwareFailed();
        return false;
    }

    //trigger settings
    QString slope = QString("POS");
    if(config.slope == BlackChirp::FallingEdge)
        slope = QString("NEG");
    QString trigCh = QString("AUX");
    if(config.trigChannel > 0)
        trigCh = QString("CHAN%1").arg(config.trigChannel);

    if(!scopeCommand(QString(":TRIGGER:SWEEP TRIGGERED")))
    {
        exp.setHardwareFailed();
        return false;
    }
    if(!scopeCommand(QString(":TRIGGER:LEVEL %1,%2").arg(trigCh).arg(config.trigLevel,0,'f',3)))
    {
        exp.setHardwareFailed();
        return false;
    }
    if(!scopeCommand(QString(":TRIGGER:MODE EDGE")))
    {
        exp.setHardwareFailed();
        return false;
    }
    if(!scopeCommand(QString(":TRIGGER:EDGE:SOURCE %1").arg(trigCh)))
    {
        exp.setHardwareFailed();
        return false;
    }
    if(!scopeCommand(QString(":TRIGGER:EDGE:SLOPE %1").arg(slope)))
    {
        exp.setHardwareFailed();
        return false;
    }


    //set trigger position to left of screen
    if(!scopeCommand(QString(":TIMEBASE:REFERENCE LEFT")))
    {
        exp.setHardwareFailed();
        return false;
    }



    //Data transfer stuff. LSBFirst is faster, and we'll use 2 bytes because averaging
    //will probably be done
    //write data transfer commands
    if(!scopeCommand(QString(":WAVEFORM:SOURCE CHAN%1").arg(config.fidChannel)))
    {
        exp.setHardwareFailed();
        return false;
    }
    if(!scopeCommand(QString(":WAVEFORM:BYTEORDER LSBFIRST")))
    {
        exp.setHardwareFailed();
        return false;
    }
    if(!scopeCommand(QString(":WAVEFORM:FORMAT WORD")))
    {
        exp.setHardwareFailed();
        return false;
    }
    if(!scopeCommand(QString(":WAVEFORM:STREAMING ON")))
    {
        exp.setHardwareFailed();
        return false;
    }
    config.byteOrder = QDataStream::BigEndian;
    config.bytesPerPoint = 2;


    //calculate y multipliers and x spacing, since this scope will not
    //update those until after waveforms have been acquired
    config.yMult = config.vScale*10.0/32768.0;
    config.xIncr = 1.0/config.sampleRate;
    config.yOff = 0;

    //now the fast frame/segmented stuff
    if(config.fastFrameEnabled)
    {
        if(!scopeCommand(QString(":ACQUIRE:MODE SEGMENTED")))
        {
            exp.setHardwareFailed();
            return false;
        }
        if(!scopeCommand(QString(":ACQUIRE:SEGMENTED:COUNT %1").arg(config.numFrames)))
        {
            exp.setHardwareFailed();
            return false;
        }
        if(!scopeCommand(QString(":WAVEFORM:SEGMENTED:ALL ON")))
        {
            exp.setHardwareFailed();
            return false;
        }
    }
    else
    {

        config.numFrames = 1;

        if(!scopeCommand(QString(":ACQUIRE:MODE RTIME")))
        {
            exp.setHardwareFailed();
            return false;
        }
    }

    //block averaging...
    if(config.blockAverageEnabled)
    {
        if(!scopeCommand(QString(":ACQUIRE:AVERAGE ON")))
        {
            exp.setHardwareFailed();
            return false;
        }
        if(!scopeCommand(QString(":ACQUIRE:COUNT %1").arg(config.numAverages)))
        {
            exp.setHardwareFailed();
            return false;
        }
        config.blockAverageMultiply = true;
    }
    else
        config.numAverages = 1;

    //sample rate and point settings
    if(!scopeCommand(QString(":ACQUIRE:SRATE:ANALOG:AUTO OFF")))
    {
        exp.setHardwareFailed();
        return false;
    }
    if(!scopeCommand(QString(":ACQUIRE:SRATE:ANALOG %1").arg(QString::number(config.sampleRate,'g',2))))
    {
        exp.setHardwareFailed();
        return false;
    }
    if(!scopeCommand(QString(":ACQUIRE:POINTS:AUTO OFF")))
    {
        exp.setHardwareFailed();
        return false;
    }
    if(!scopeCommand(QString(":ACQUIRE:POINTS:ANALOG %1").arg(config.recordLength)))
    {
        exp.setHardwareFailed();
        return false;
    }


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
    if(resp.isEmpty() || !resp.contains(QString("CHAN%1").arg(config.fidChannel).toLatin1()))
    {
        emit logMessage(QString("Failed to set FID channel. Response to waveform source query: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),BlackChirp::LogError);
        exp.setHardwareFailed();
        return false;
    }

    //read actual offset and vertical scale
    resp = p_comm->queryCmd(QString(":CHAN%1:OFFSET?\n").arg(config.fidChannel));
    if(!resp.isEmpty())
    {
        bool ok = false;
        double offset = resp.trimmed().toDouble(&ok);
        if(!ok)
        {
            emit logMessage(QString("Could not parse offset response. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
        config.vOffset = offset;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to offset query."),BlackChirp::LogError);
        exp.setHardwareFailed();
        return false;
    }
    resp = p_comm->queryCmd(QString(":CHAN%1:SCALE?\n").arg(config.fidChannel));
    if(!resp.isEmpty())
    {
        bool ok = false;
        double scale = resp.trimmed().toDouble(&ok);
        if(!ok)
        {
            emit logMessage(QString("Could not parse scale response. Response: %2 (Hex: %3)").arg(QString(resp)).arg(QString(resp.toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
        if(!(fabs(config.vScale-scale) < 0.01))
            emit logMessage(QString("Vertical scale is different than specified. Target: %1 V/div, Scope setting: %2 V/div").arg(QString::number(config.vScale,'f',3))
                            .arg(QString::number(scale,'f',3)),BlackChirp::LogWarning);
        config.vScale = scale;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to scale query."),BlackChirp::LogError);
        exp.setHardwareFailed();
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
            emit logMessage(QString("Sample rate query returned an invalid response. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
        if(!(fabs(sRate - config.sampleRate)<1e6))
        {
            emit logMessage(QString("Could not set sample rate successfully. Target: %1 GS/s, Scope setting: %2 GS/s").arg(QString::number(config.sampleRate/1e9,'f',3))
                            .arg(QString::number(sRate/1e9,'f',3)),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
        config.sampleRate = sRate;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to sample rate query."),BlackChirp::LogError);
        exp.setHardwareFailed();
        return false;
    }

    resp = p_comm->queryCmd(QString(":ACQUIRE:POINTS?\n"));
    if(!resp.isEmpty())
    {
        bool ok = false;
        int recLength = resp.trimmed().toInt(&ok);
        if(!ok)
        {
            emit logMessage(QString("Record length query returned an invalid response. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
        if(!(abs(recLength-config.recordLength) < 1000))
        {
            emit logMessage(QString("Record length limited by scope memory. Length will be different than requested. Target: %1, Scope setting: %2").arg(QString::number(config.recordLength))
                            .arg(QString::number(recLength)),BlackChirp::LogWarning);
        }
        config.recordLength = recLength;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to record length query."),BlackChirp::LogError);
        exp.setHardwareFailed();
        return false;
    }

    resp = p_comm->queryCmd(QString(":TRIGGER:EDGE:SOURCE?\n"));
    if(resp.isEmpty() || !QString(resp).contains(trigCh),Qt::CaseInsensitive)
    {
        emit logMessage(QString("Could not verify trigger channel. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),BlackChirp::LogError);
        exp.setHardwareFailed();
        return false;
    }


    resp = p_comm->queryCmd(QString(":TRIGGER:EDGE:SLOPE?\n"));
    if(resp.isEmpty() || !QString(resp).contains(slope))
    {
        emit logMessage(QString("Could not verify trigger slope. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),BlackChirp::LogError);
        exp.setHardwareFailed();
        return false;
    }

    d_configuration = config;
    exp.setScopeConfig(config);
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
    qint64 bytes = d_configuration.bytesPerPoint*d_configuration.recordLength*d_configuration.numFrames;

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
        emit logMessage(QString("Timed out on query %1").arg(orig),BlackChirp::LogError);
        return false;
    }

    int val = resp.trimmed().toInt();
    if(val != 0)
    {
        emit logMessage(QString("Received error %1 on query %2").arg(val).arg(orig),BlackChirp::LogError);
        return false;
    }
    return true;
}
