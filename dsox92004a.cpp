#include "dsox92004a.h"

#include <QTcpSocket>

DSOx92004A::DSOx92004A(QObject *parent) : FtmwScope(parent)
{
    d_subKey = QString("DSOx92004A");
    d_prettyName = QString("Ftmw Oscilloscope DSOx92004A");
    d_commType = CommunicationProtocol::Tcp;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    s.setValue(QString("canBlockAverage"),false);
    s.setValue(QString("canFastFrame"),true);
    s.setValue(QString("canSummaryFrame"),false);
    s.setValue(QString("canBlockAndFastFrame"),true);

    double bandwidth = s.value(QString("bandwidth"),20000.0).toDouble();
    s.setValue(QString("bandwidth"),bandwidth);

    if(s.beginReadArray(QString("sampleRates")) > 0)
        s.endArray();
    else
    {
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
    if(!p_comm->testConnection())
    {
        emit connected(false);
        return false;
    }

    QByteArray resp = p_comm->queryCmd(QString("*IDN?\n"));

    if(resp.isEmpty())
    {
        emit connected(false,QString("Did not respond to ID query."));
        return false;
    }

    if(resp.length() > 100)
        resp = resp.mid(0,100);

    if(!resp.startsWith(QByteArray("Keysight Technologies,DSO")))
    {
        emit connected(false,QString("ID response invalid. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp)));
    emit connected();
    return true;

}

void DSOx92004A::initialize()
{
    p_comm->initialize();
    p_comm->setReadOptions(1000,true,QByteArray("\n"));
    p_socket = dynamic_cast<QTcpSocket*>(p_comm->device());
    connect(p_socket,static_cast<void (QTcpSocket::*)(QAbstractSocket::SocketError)>(&QTcpSocket::error),this,&MSO72004C::socketError);
    p_socket->setSocketOption(QAbstractSocket::LowDelayOption,1);

    p_queryTimer = new QTimer(this);
    connect(p_queryTimer,&QTimer::timeout,this,&DSOx92004A::readWaveform);
    testConnection();
}

Experiment DSOx92004A::prepareForExperiment(Experiment exp)
{
    d_enabledForExperiment = exp.ftmwConfig().isEnabled();
    if(!d_enabledForExperiment)
        return exp;

    auto config = exp.ftmwConfig().scopeConfig();

    //disable ugly headers
    if(!p_comm->writeCmd(QString(":SYSTEM:HEADER OFF\n")))
    {
        emit logMessage(QString("Could not disable verbose header mode."),BlackChirp::LogError);
        exp.setHardwareFailed();
        return exp;
    }

    //write data transfer commands
    if(!p_comm->writeCmd(QString(":WAVEFORM:SOURCE CHAN%\n").arg(config.fidChannel)))
    {
        emit logMessage(QString("Could not set waveform source. Write failed."),BlackChirp::LogError);
        exp.setHardwareFailed();
        return exp;
    }

    //clear out socket before senting our first query
    if(p_socket->bytesAvailable())
        p_socket->readAll();

    //verify that FID channel was set correctly
    QByteArray resp = p_comm->queryCmd(QString(":WAVEFORM:SOURCE?\n"));
    if(resp.isEmpty() || !resp.contains(QString("CHAN%1").arg(config.fidChannel).toLatin1()))
    {
        emit logMessage(QString("Failed to set FID channel. Response to waveform source query: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),BlackChirp::LogError);
        exp.setHardwareFailed();
        return exp;
    }

    if(!p_comm->writeCmd(QString(":CHAN%1:BWLIMIT OFF; INPUT DC50;OFFSET 0;SCALE %2\n")
                         .arg(config.fidChannel).arg(QString::number(config.vScale,'e',3))))
    {
        emit logMessage(QString("Failed to write channel settings."),BlackChirp::LogError);
        exp.setHardwareFailed();
        return exp;
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
            return exp;
        }
        config.vOffset = offset;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to offset query."),BlackChirp::LogError);
        exp.setHardwareFailed();
        return exp;
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
            return exp;
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
        return exp;
    }

    //sample rate and point settings
    if(!p_comm->writeCmd(QString(":ACQUIRE:SRATE %1;POINTS %2\n")
                         .arg(QString::number(config.sampleRate,'g',6)).arg(config.recordLength)))
    {
        emit logMessage(QString("Could not apply sample rate/point settings."),BlackChirp::LogError);
        exp.setHardwareFailed();
        return exp;
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
            return exp;
        }
        if(!(fabs(sRate - config.sampleRate)<1e6))
        {
            emit logMessage(QString("Could not set sample rate successfully. Target: %1 GS/s, Scope setting: %2 GS/s").arg(QString::number(config.sampleRate/1e9,'f',3))
                            .arg(QString::number(sRate/1e9,'f',3)),BlackChirp::LogError);
            exp.setHardwareFailed();
            return exp;
        }
        config.sampleRate = sRate;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to sample rate query."),BlackChirp::LogError);
        exp.setHardwareFailed();
        return exp;
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
            return exp;
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
        return exp;
    }

    //trigger settings
    QString slope = QString("POS");
    if(config.slope == BlackChirp::FallingEdge)
        slope = QString("NEG");
    QString trigCh = QString("AUX");
    if(config.trigChannel > 0)
        trigCh = QString("CHAN%1").arg(config.trigChannel);

    resp = p_comm->queryCmd(QString(":TRIGGER:MODE EDGE;:TRIGGER:EDGE:SOURCE %1;COUPLING DC;SLOPE %2;:TRIGGER:LEVEL %3;:TRIGGER:EDGE:SOURCE?;:TRIGGER:SLOPE?\n").arg(trigCh).arg(slope).arg(config.trigLevel,0,'f',3));
    if(!resp.isEmpty())
    {
        if(!QString(resp).contains(trigCh),Qt::CaseInsensitive)
        {
            emit logMessage(QString("Could not verify trigger channel. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return exp;
        }

        if(!QString(resp).contains(slope,Qt::CaseInsensitive))
        {
            emit logMessage(QString("Could not verify trigger slope. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return exp;
        }
    }
    else
    {
        emit logMessage(QString("Gave an empty response to trigger query."),BlackChirp::LogError);
        exp.setHardwareFailed();
        return exp;
    }

    //Data transfer stuff. LSBFirst is faster, and we'll use 2 bytes because averaging
    //will probably be done
    p_comm->writeCmd(QString(":WAVEFORM:BYTEORDER LSBFIRST;FORMAT WORD;STREAMING ON\n"));
    config.byteOrder = QDataStream::BigEndian;
    config.bytesPerPoint = 2;


    //calculate y multipliers and x spacing, since this scope will not
    //update those until after waveforms have been acquired
    config.yMult = config.vScale*10.0/32768.0;
    config.xIncr = 1,0/config.sampleRate;
    config.yOff = 0;

    //now the fast frame/segmented stuff
    if(config.fastFrameEnabled)
    {
        resp = p_comm->queryCmd(QString(":ACQUIRE:MODE SEGMENTED;ACQUIRE:SEGMENTED:COUNT %1;COUNT?\n").arg(config.numFrames));
        if(resp.isEmpty())
        {
            emit logMessage(QString("Gave an empty response to segmented frame count query."),BlackChirp::LogError);
            exp.setHardwareFailed();
            return exp;
        }
        int nf = resp.trimmed().toInt();
        if(nf != config.numFrames)
        {
            emit logMessage(QString("Could not set number of segmented frames to desired value. Requested: %1, Actual: %2").arg(config.numFrames).arg(nf),BlackChirp::LogError);
            exp.setHardwareFailed();
            return exp;
        }
    }
    else
    {
        p_comm->writeCmd(QString(":ACQUIRE:MODE RTIME\n"));
        config.numFrames = 1;
    }

    //block averaging...
    if(config.blockAverageEnabled)
    {
        resp = p_comm->queryCmd(QString(":ACQUIRE:AVERAGE ON;COUNT %1;COUNT?\n").arg(config.numAverages));
        if(resp.isEmpty())
        {
            emit logMessage(QString("Gave an empty response to average count query."),BlackChirp::LogError);
            exp.setHardwareFailed();
            return exp;
        }
        int na = resp.trimmed().toInt();
        if(na != config.numAverages)
        {
            emit logMessage(QString("Could not set number of averages to desired value. Requested: %1, Actual: %2").arg(config.numFrames).arg(nf),BlackChirp::LogError);
            exp.setHardwareFailed();
            return exp;
        }
    }

    d_configuration = config;
    exp.setScopeConfig(config);
    d_acquiring = false;

    return exp;


}

void DSOx92004A::beginAcquisition()
{
    if(d_enabledForExperiment)
    {
        p_comm->writeCmd(QString(":DIGITIZE\n"));
        p_queryTimer->start(100);
    }
}

void DSOx92004A::endAcquisition()
{
    if(d_enabledForExperiment)
    {
        disconnect(p_socket, &QTcpSocket::readyRead, this, &DSOx92004A::retrieveData);
        p_queryTimer->stop();
        p_comm->writeCmd(QString("*CLS\n"));
    }
}

void DSOx92004A::readTimeData()
{
}

void DSOx92004A::readWaveform()
{
    QByteArray resp = p_comm->queryCmd(QString("*OPC?\n"));
    if(resp.contains('1'))
    {
        //begin next transfer -- TEST
//        p_comm->writeCmd(QString(":DIGITIZE\n"));

        //grab waveform data directly from socket;
        p_queryTimer->stop();
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
    p_comm->writeCmd(QString(":DIGITIZE\n"));
    p_queryTimer->start(100);

}
