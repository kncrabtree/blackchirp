#include "dsa71604c.h"

#include <QTcpSocket>
#include <QTimer>
#include <QThread>
#include <math.h>

using namespace BC::Key::FtmwScope;
using namespace BC::Key::Digi;

Dsa71604c::Dsa71604c(QObject *parent) :
    FtmwScope(dsa71604c,dsa71064cName,CommunicationProtocol::Tcp,parent),
    d_waitingForReply(false), d_foundHeader(false),
    d_headerNumBytes(0), d_waveformBytes(0)
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
    setDefault(bandwidth,16000.0);

    if(!containsArray(sampleRates))
        setArray(sampleRates,{
                     {{srText,"2 GSa/s"},{srValue,2e9}},
                       {{srText,"5 GSa/s"},{srValue,5e9}},
                       {{srText,"10 GSa/s"},{srValue,10e9}},
                       {{srText,"20 GSa/s"},{srValue,20e9}},
                       {{srText,"50 GSa/s"},{srValue,50e9}},
                       {{srText,"100 GSa/s"},{srValue,100e9}}
                     });

    save();
}

Dsa71604c::~Dsa71604c()
{

}

bool Dsa71604c::testConnection()
{

    p_comm->writeCmd(QString("*CLS\n"));
    p_comm->writeCmd(QString("*CLS\n"));
    QByteArray resp = scopeQueryCmd(QString("*IDN?\n"));

    if(resp.isEmpty())
    {
        d_errorString = QString("Did not respond to ID query.");
        return false;
    }

    if(!resp.startsWith(QByteArray("TEKTRONIX,DSA71604C")))
    {
        if(resp.length() > 50)
            resp = resp.mid(0,50);
        d_errorString = QString("ID response invalid. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex()));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp)));
    return true;
}

void Dsa71604c::initialize()
{
    p_scopeTimeout = new QTimer(this);

    p_comm->setReadOptions(3000,true,QByteArray("\n"));
    p_socket = dynamic_cast<QTcpSocket*>(p_comm->device());
    connect(p_socket,static_cast<void (QTcpSocket::*)(QAbstractSocket::SocketError)>(&QTcpSocket::errorOccurred),this,&Dsa71604c::socketError);
    p_socket->setSocketOption(QAbstractSocket::LowDelayOption,1);
    p_socket->setSocketOption(QAbstractSocket::KeepAliveOption,1);
}

bool Dsa71604c::prepareForExperiment(Experiment &exp)
{
    //attempt to apply settings. return invalid configuration if anything fails.
    //this is a lot of really tedious code.
    //All settings need to be made, and most of them also need to be verified
    //error messages are spit out to the UI
    //If this frequently fails, I recommend turning verbose headers on and writing a custom query command that verifies the header response, retrying until a valid reply is received.

    //make a copy of the configuration in which to store settings
    d_enabledForExperiment = exp.ftmwEnabled();
    if(!d_enabledForExperiment)
        return true;

    auto &config = exp.ftmwConfig()->d_scopeConfig;

    disconnect(p_socket,&QTcpSocket::readyRead,this,&Dsa71604c::readWaveform);

    //disable ugly headers
    if(!p_comm->writeCmd(QString(":HEADER OFF\n")))
    {
        emit logMessage(QString("Could not disable verbose header mode."),LogHandler::Error);
        return false;
    }

    //write data transfer commands
    if(!p_comm->writeCmd(QString(":DATA:SOURCE CH%1;START 1;STOP 200000\n").arg(config.d_fidChannel)))
    {
        emit logMessage(QString("Could not write :DATA commands."),LogHandler::Error);
        return false;
    }

    //clear out socket before senting our first query
    if(p_socket->bytesAvailable())
        p_socket->readAll();

    //verify that FID channel was set correctly
    QByteArray resp = scopeQueryCmd(QString(":DATA:SOURCE?\n"));
    if(resp.isEmpty() || !resp.contains(QString("CH%1").arg(config.d_fidChannel).toLatin1()))
    {
        emit logMessage(QString("Failed to set FID channel. Response to data source query: %1 (Hex: %2)").
                        arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
        return false;
    }

//    if(!d_comm->writeCmd(QString("CH%1:BANDWIDTH:ENHANCED OFF; CH%1:BANDWIDTH 1.6+10; COUPLING AC;OFFSET 0;SCALE %2\n").arg(config.fidChannel).arg(QString::number(config.vScale,'g',4))))
    if(!p_comm->writeCmd(QString(":CH%1:BANDWIDTH FULL; COUPLING AC;OFFSET %2;SCALE %3\n")
                         .arg(config.d_fidChannel)
                         .arg(QString::number(config.d_analogChannels[config.d_fidChannel].offset,'g',4))
                         .arg(QString::number(config.d_analogChannels[config.d_fidChannel].fullScale/5.0,'g',4))))
    {
        emit logMessage(QString("Failed to write channel settings."),LogHandler::Error);
        return false;
    }

    //read actual offset and vertical scale
    resp = scopeQueryCmd(QString(":CH%1:OFFSET?\n").arg(config.d_fidChannel));
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
        config.d_analogChannels[config.d_fidChannel].offset = offset;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to offset query."),LogHandler::Error);
        return false;
    }

    resp = scopeQueryCmd(QString(":CH%1:SCALE?\n").arg(config.d_fidChannel));
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
        if(!(fabs(config.d_analogChannels[config.d_fidChannel].fullScale/5.0-scale) < 0.01))
            emit logMessage(QString("Vertical full scale is different than specified. Target: %1 V, Scope setting: %2 V")
                            .arg(QString::number(config.d_analogChannels[config.d_fidChannel].fullScale/5.0,'f',3))
                            .arg(QString::number(scale*5.0,'f',3)),LogHandler::Warning);
        config.d_analogChannels[config.d_fidChannel].fullScale = scale*5.0;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to scale query."),LogHandler::Error);
        return false;
    }

    //horizontal settings
    if(!p_comm->writeCmd(QString(":HORIZONTAL:MODE MANUAL;:HORIZONTAL:DELAY:MODE ON;:HORIZONTAL:DELAY:POSITION 0;:HORIZONTAL:DELAY:TIME %1;:HORIZONTAL:MODE:SAMPLERATE %2;RECORDLENGTH %3\n")
                         .arg(QString::number(config.d_triggerDelayUSec,'g',6)).arg(QString::number(config.d_sampleRate,'g',6)).arg(config.d_recordLength)))
    {
        emit logMessage(QString("Could not apply horizontal settings."),LogHandler::Error);
        return false;
    }

    //verify sample rate, record length, and horizontal delay
    resp = scopeQueryCmd(QString(":HORIZONTAL:MODE:SAMPLERATE?\n"));
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
    resp = scopeQueryCmd(QString(":HORIZONTAL:MODE:RECORDLENGTH?\n"));
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
            emit logMessage(QString("Could not set record length successfully! Target: %1, Scope setting: %2")
                            .arg(QString::number(config.d_recordLength))
                            .arg(QString::number(recLength)),LogHandler::Error); 
            return false;
        }
        config.d_recordLength = recLength;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to record length query."),LogHandler::Error);
        return false;
    }
    resp = scopeQueryCmd(QString(":HORIZONTAL:DELAY:TIME?\n"));
    if(!resp.isEmpty())
    {
        bool ok = false;
        double delay = resp.trimmed().toDouble(&ok);
        if(!ok)
        {
            emit logMessage(QString("Trigger delay query returned an invalid response. Response: %1 (Hex: %2)")
                            .arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
            return false;
        }
        if(fabs(delay-config.d_triggerDelayUSec) > 1e-6)
        {
            emit logMessage(QString("Could not set trigger delay successfully! Target: %1, Scope setting: %2")
                            .arg(QString::number(config.d_triggerDelayUSec))
                            .arg(QString::number(delay)),LogHandler::Error);    
            return false;
        }
        config.d_triggerDelayUSec = delay;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to trigger delay query."),LogHandler::Error);
        return false;
    }

    //fast frame settings
    if(!config.d_multiRecord && !config.d_blockAverage)
    {
        resp = scopeQueryCmd(QString(":HORIZONTAL:FASTFRAME:STATE OFF;STATE?\n"));
        if(!resp.isEmpty())
        {
            bool ok = false;
            bool ffState = (bool)resp.trimmed().toInt(&ok);
            if(!ok || ffState)
            {
                emit logMessage(QString("Could not disable FastFrame mode."),LogHandler::Error);    
                return false;
            }
        }
        else
        {
            emit logMessage(QString("Gave an empty response to FastFrame state query."),LogHandler::Error);
            return false;
        }
    }
    else
    {
        //enable fastframe and disable summary frame; verify
        resp = scopeQueryCmd(QString(":HORIZONTAL:FASTFRAME:STATE ON;SUMFRAME NON;STATE?\n"));
        if(!resp.isEmpty())
        {
            bool ok = false;
            bool ffState = (bool)resp.trimmed().toInt(&ok);
            if(!ok || !ffState)
            {
                emit logMessage(QString("Could not enable FastFrame mode."),LogHandler::Error);
                return false;
            }
        }
        else
        {
            emit logMessage(QString("Gave an empty response to FastFrame state query."),LogHandler::Error);
            return false;
        }

        //now, check max number of frames
        resp = scopeQueryCmd(QString(":HORIZONTAL:FASTFRAME:MAXFRAMES?\n"));
        if(!resp.isEmpty())
        {
            bool ok = false;
            int maxFrames = resp.trimmed().toInt(&ok);
            if(!ok || maxFrames < 1)
            {
                emit logMessage(QString("Could not determine maximum number of frames in FastFrame mode."),LogHandler::Error);
                return false;
            }

            int numFrames = qMax(config.d_numRecords,config.d_numAverages);
            if(config.d_blockAverage)
                maxFrames -= 2; //note: if summary frame enabled, need to reserve 2 frames worth of memory

            if(maxFrames < numFrames)
            {
                emit logMessage(QString("Requested number of Fast frames (%1) is greater than maximum possible value with the requested acquisition settings (%2).")
                                .arg(numFrames).arg(maxFrames),LogHandler::Error);
                return false;
            }

            resp = scopeQueryCmd(QString(":HORIZONTAL:FASTFRAME:COUNT %1;COUNT?\n").arg(numFrames));
            if(!resp.isEmpty())
            {
                ok = false;
                int n = resp.trimmed().toInt(&ok);
                if(!ok)
                {
                    emit logMessage(QString("FastFrame count query returned an invalid response. Response: %1 (Hex: %2)")
                                    .arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
                    return false;
                }
                if(n != numFrames)
                {
                    emit logMessage(QString("Requested number of FastFrames (%1) is different than actual number (%2).").arg(numFrames).arg(n));
                    return false;
                }
            }
            else
            {
                emit logMessage(QString("Gave an empty response to FastFrame count query."),LogHandler::Error);
                return false;
            }

            QString sumfConfig = QString("AVE");
            if(!config.d_blockAverage)
                sumfConfig = QString("NON");
            resp = scopeQueryCmd(QString(":HORIZONTAL:FASTFRAME:SUMFRAME %1;SUMFRAME?\n").arg(sumfConfig));
            if(!resp.isEmpty())
            {
                if(!QString(resp).contains(sumfConfig,Qt::CaseInsensitive))
                {
                    emit logMessage(QString("Could not configure FastFrame summary frame to %1. Response: %2 (Hex: %3)")
                                    .arg(sumfConfig).arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
                    return false;
                }
            }
            else
            {
                emit logMessage(QString("Gave an empty response to FastFrame summary frame query."),LogHandler::Error);
                return false;
            }

            if(config.d_blockAverage)
            {
                //this forces the scope to only return the final frame, which is the summary frame
                if(!p_comm->writeCmd(QString(":DATA:FRAMESTART 100000;FRAMESTOP 100000\n")))
                {
                    emit logMessage(QString("Could not configure summary frame."),LogHandler::Error);
                    return false;
                }
            }
            else
            {
                //this forces the scope to return all frames
                if(!p_comm->writeCmd(QString(":DATA:FRAMESTART 1;FRAMESTOP 100000\n")))
                {
                    emit logMessage(QString("Could not configure frames."),LogHandler::Error);
                    return false;
                }
            }
        }
        else
        {
            emit logMessage(QString("Gave an empty response to FastFrame max frames query."),LogHandler::Error);
            return false;
        }
    }

    //trigger settings
    QString slope = QString("RIS");
    if(config.d_triggerSlope == FallingEdge)
        slope = QString("FALL");
    QString trigCh = QString("AUX");
    if(config.d_triggerChannel > 0)
        trigCh = QString("CH%1").arg(config.d_triggerChannel);
    resp = scopeQueryCmd(QString(":TRIGGER:A:EDGE:SOURCE %1;COUPLING DC;SLOPE %2;:TRIGGER:A:LEVEL %3;:TRIGGER:A:EDGE:SOURCE?;SLOPE?\n")
                         .arg(trigCh).arg(slope).arg(config.d_triggerLevel,0,'f',3));
    if(!resp.isEmpty())
    {
        if(!QString(resp).contains(trigCh,Qt::CaseInsensitive))
        {
            emit logMessage(QString("Could not verify trigger channel. Response: %1 (Hex: %2)")
                            .arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
            return false;
        }

        if(!QString(resp).contains(slope,Qt::CaseInsensitive))
        {
            emit logMessage(QString("Could not verify trigger slope. Response: %1 (Hex: %2)")
                            .arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
            return false;
        }
    }
    else
    {
        emit logMessage(QString("Gave an empty response to trigger query."),LogHandler::Error);
        return false;
    }

    //set waveform output settings
    if(config.d_blockAverage)
    {
        if(config.d_bytesPerPoint != 2)
            emit logMessage("Settting bytes per point to 2 for averaging",LogHandler::Warning);
        config.d_bytesPerPoint = 2;
        resp = p_comm->queryCmd(QString(":HORIZONTAL:FASTFRAME:SIXTEENBIT ON;SIXTEENBIT?\n"));
        if(!resp.contains("1"))
        {
            emit logMessage("Could not configure scope for 16-bit summary frame",LogHandler::Error);
            return false;
        }
    }
    else
        p_comm->writeCmd(QString(":HORIZONTAL:FASTFRAME:SIXTEENBIT OFF\n"));

    if(!p_comm->writeCmd(QString(":WFMOUTPRE:ENCDG BIN;BN_FMT RI;BYT_OR LSB;BYT_NR %1\n").arg(config.d_bytesPerPoint)))
    {
        emit logMessage(QString("Could not send waveform output commands."),LogHandler::Error);
        return false;
    }

    //acquisition settings
    if(!p_comm->writeCmd(QString(":ACQUIRE:MODE SAMPLE;STOPAFTER RUNSTOP;STATE RUN\n")))
    {
        emit logMessage(QString("Could not send acquisition commands."),LogHandler::Error);
        return false;
    }

    //force a trigger event to update these settings
    if(config.d_multiRecord || config.d_blockAverage)
    {
        auto numFrames = qMax(config.d_numRecords,config.d_numAverages);
        for(int i=0;i<numFrames;i++)
        {
            p_comm->writeCmd(QString(":TRIGGER FORCE\n"));
            QThread::msleep(2);
        }
    }
    else
    {
        p_comm->writeCmd(QString(":TRIGGER FORCE\n"));
        QThread::msleep(2);
    }

    //read certain output settings from scope
    resp = scopeQueryCmd(QString(":WFMOUTPRE:ENCDG?;BN_FMT?;BYT_OR?;NR_FR?;NR_PT?;BYT_NR?\n"));
    if(!resp.isEmpty())
    {
        QStringList l = QString(resp.trimmed()).split(QChar(';'),Qt::SkipEmptyParts);
        if(l.size() < 6)
        {
            emit logMessage(QString("Could not parse response to waveform output settings query. Response: %1 (Hex: %2)")
                            .arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
            return false;
        }

        //check encoding
        if(!l.at(0).contains(QString("BIN"),Qt::CaseInsensitive))
        {
            emit logMessage(QString("Waveform encoding could not be set to binary. Response: %1 (Hex: %2)")
                            .arg(l.at(0)).arg(QString(l.at(0).toLatin1().toHex())),LogHandler::Error);  
            return false;
        }
        //check binary format
        if(!l.at(1).contains(QString("RI"),Qt::CaseInsensitive))
        {
            emit logMessage(QString("Waveform format could not be set to signed integer. Response: %1 (Hex: %2)")
                            .arg(l.at(1)).arg(QString(l.at(1).toLatin1().toHex())),LogHandler::Error);
            return false;
        }
        //check byte order
        if(!l.at(2).contains(QString("LSB"),Qt::CaseInsensitive))
        {
            emit logMessage(QString("Waveform format could not be set to least significant byte first. Response: %1 (Hex: %2)")
                            .arg(l.at(2)).arg(QString(l.at(2).toLatin1().toHex())),LogHandler::Error);
            return false;
        }
        config.d_byteOrder = DigitizerConfig::LittleEndian;

        //verify number of frames
        if(config.d_multiRecord)
        {
            if(l.at(3).toInt() != config.d_numRecords)
            {
                emit logMessage(QString("Waveform contains the wrong number of frames. Target: %1, Actual: %2. Response: %3 (Hex: %4)")
                                .arg(config.d_numRecords).arg(l.at(3).toInt()).arg(l.at(3)).arg(QString(l.at(3).toLatin1().toHex())),LogHandler::Error);
                return false;
            }
        }
        else if (l.at(3).toInt() != 1)
        {
            emit logMessage(QString("Waveform contains the wrong number of frames. Target: 1. Actual: %1. Response: %2 (Hex: %3)")
                            .arg(l.at(3).toInt()).arg(l.at(3)).arg(QString(l.at(3).toLatin1().toHex())),LogHandler::Error);
            return false;
        }

        //verify record length
        bool ok = false;
        int recLen = l.at(4).toInt(&ok);
        if(!ok)
        {
            emit logMessage(QString("Could not parse waveform record length response. Response: %1 (Hex: %2)")
                            .arg(l.at(4)).arg(QString(l.at(4).toLatin1().toHex())),LogHandler::Error);
            return false;
        }
        if(recLen != config.d_recordLength)
        {
            emit logMessage(QString("Record length is %1. Requested value was %2. Proceeding with %1 samples.")
                            .arg(recLen).arg(config.d_recordLength),LogHandler::Warning);
            config.d_recordLength = recLen;
        }
        //verify byte number
        int bpp = l.at(5).mid(0,1).toInt(&ok);
        if(!ok || bpp != config.d_bytesPerPoint)
        {
            emit logMessage(QString("Invalid response to bytes per point query. Response: %1 (Hex: %2)")
                            .arg(l.at(8)).arg(QString(l.at(8).toLatin1().toHex())),LogHandler::Error);
            return false;
        }
    }

    auto cfg = dynamic_cast<FtmwDigitizerConfig*>(this);
    if(cfg)
        *cfg = config;
    else
    {
        emit logMessage("Could not record digitizer config settings",LogHandler::Error);
        return false;
    }

    if(p_socket->bytesAvailable())
        p_socket->readAll();

    return true;
}

void Dsa71604c::beginAcquisition()
{
    if(d_enabledForExperiment)
    {
        p_comm->writeCmd(QString(":LOCK ALL;:DISPLAY:WAVEFORM OFF\n"));
        if(p_socket->bytesAvailable())
            p_socket->readAll();

        connect(p_socket,&QTcpSocket::readyRead,this,&Dsa71604c::readWaveform,Qt::UniqueConnection);
        p_comm->writeCmd(QString(":CURVESTREAM?\n"));

        d_waitingForReply = true;
        d_foundHeader = false;
        d_headerNumBytes = 0;
        d_waveformBytes = 0;
        connect(p_scopeTimeout,&QTimer::timeout,this,&Dsa71604c::wakeUp,Qt::UniqueConnection);
    }
}

void Dsa71604c::endAcquisition()
{

    if(d_enabledForExperiment)
    {
        //stop parsing waveforms
        disconnect(p_socket,&QTcpSocket::readyRead,this,&Dsa71604c::readWaveform);
        disconnect(p_scopeTimeout,&QTimer::timeout,this,&Dsa71604c::wakeUp);

        if(p_socket->bytesAvailable())
            p_socket->readAll();

        //send *CLS command twice to kick scope out of curvestream mode and clear the error queue
        p_comm->writeCmd(QString("*CLS\n"));
        p_comm->writeCmd(QString("*CLS\n"));

        if(p_socket->bytesAvailable())
            p_socket->readAll();

        p_comm->writeCmd(QString(":UNLOCK ALL;:DISPLAY:WAVEFORM ON\n"));

        d_waitingForReply = false;
        d_foundHeader = false;
        d_headerNumBytes = 0;
        d_waveformBytes = 0;
    }
}

void Dsa71604c::readWaveform()
{
    if(!d_waitingForReply) // if for some reason the readyread signal weren't disconnected, don't eat all the bytes
        return;

    qint64 ba = p_socket->bytesAvailable();
//    emit logMessage(QString("Bytes available: %1\t%2 ms").arg(ba).arg(QTime::currentTime().msec()));

    //waveforms are returned from the scope in the format #xyyyyyyy<data>\n
    //the reply starts with the '#' character
    //the next byte (x) is a hex digit that tells how many bytes of header information follow
    //yyyyyy is an ASCII sequence of x numbers that tells how many data bytes the waveform contains
    //<data> is the actual data block
    //the final character is an ASCII newline

    if(!d_foundHeader) //here, we start looking for a block data header, which starts with '#'
    {
//        emit consoleMessage(tr("Looking for header, BA %1, %2  %3").arg(ba).arg(scope->peek(ba).toHex().constData()).arg(scope->peek(ba).constData()));
//        emit consoleMessage(tr("Looking for header, BA %1").arg(ba));
        char c=0;
        qint64 i=0;
        while(i<ba && !d_foundHeader)
        {
            p_socket->getChar(&c);
            if(c=='#')
            {
                d_foundHeader = true;
                p_scopeTimeout->stop();
                p_scopeTimeout->start(600000);
//                emit logMessage(QString("Found hdr: %1 ms").arg(QTime::currentTime().msec()));
            }
            i++;
        }
    }

    if(d_foundHeader && d_headerNumBytes == 0) //we've found the header hash, now get the number of header bytes
    {
        //make sure the entire header can be read
        if(p_socket->bytesAvailable())
        {
            //there is a header byte ready. Read it and set the number of remaining header bytes
            char c=0;
            p_socket->getChar(&c);
            QString hdrNum = QString::fromLatin1(&c,1);
            bool ok = false;
            int nb = hdrNum.toInt(&ok,16);
            if(!ok || nb < 1 || nb > 15)
            {
                //it's possible that we're in the middle of an old waveform by fluke.
                //continue looking for '#'
                d_foundHeader = false;
                if(p_socket->bytesAvailable())
                    readWaveform();

                return;
            }
            d_headerNumBytes = nb;
        }
        else
            return;
    }

    if(d_foundHeader && d_headerNumBytes > 0 && d_waveformBytes == 0) //header hash and number of header bytes read, need to read # wfm bytes
    {
        if(p_socket->bytesAvailable() >= d_headerNumBytes)
        {
            QByteArray wfmBytes = p_socket->read(d_headerNumBytes);
            bool ok = false;
            int b = wfmBytes.toInt(&ok);
            int nf = 1;
            if(d_multiRecord)
                nf = d_numRecords;
            if(!ok || b != d_recordLength*d_bytesPerPoint*nf)
            {
                //it's possible that we're in the middle of an old waveform by fluke.
                //continue looking for '#'
                d_foundHeader = false;
                d_headerNumBytes = 0;
                if(p_socket->bytesAvailable())
                    readWaveform();

                return;
            }
            d_waveformBytes = b;
        }
        else
            return;
    }

    if(d_foundHeader && d_headerNumBytes > 0 && d_waveformBytes > 0) // waiting to read waveform data
    {
        if(p_socket->bytesAvailable() >= d_waveformBytes) // whole waveform can be read!
        {
            QByteArray wfm = p_socket->read(d_waveformBytes);
//            emit logMessage(QString("Wfm read complete: %1 ms").arg(QTime::currentTime().msec()));
            emit shotAcquired(wfm);
            d_foundHeader = false;
            d_headerNumBytes = 0;
            d_waveformBytes = 0;
            return;
        }
        else
            return;
    }
}

void Dsa71604c::wakeUp()
{
    p_scopeTimeout->stop();
    emit logMessage(QString("Attempting to wake up scope"),LogHandler::Warning);

    endAcquisition();

    if(!testConnection())
    {
        emit hardwareFailure();
        return;
    }

    p_comm->writeCmd(QString(":LOCK ALL;:DISPLAY:WAVEFORM OFF\n"));
    beginAcquisition();
}

void Dsa71604c::socketError(QAbstractSocket::SocketError e)
{
    emit logMessage(QString("Socket error: %1").arg((int)e),LogHandler::Error);
    emit logMessage(QString("Error message: %1").arg(p_socket->errorString()),LogHandler::Error);
    emit hardwareFailure();
}

QByteArray Dsa71604c::scopeQueryCmd(QString query)
{
    //the scope is flaky. Sometimes, for no apparent reason, it doesn't respond
    //This will retry the query if it fails, suppressing any errors on the first try

    blockSignals(true);
    p_comm->blockSignals(true);
    QByteArray resp = p_comm->queryCmd(query);
    p_comm->blockSignals(false);
    blockSignals(false);

    if(resp.isEmpty())
        resp = p_comm->queryCmd(query);

    return resp;
}
