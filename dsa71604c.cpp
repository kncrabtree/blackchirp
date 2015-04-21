#include "dsa71604c.h"
#include "tcpinstrument.h"

Dsa71604c::Dsa71604c(QObject *parent) :
    FtmwScope(parent), d_waitingForReply(false), d_foundHeader(false),
    d_headerNumBytes(0), d_waveformBytes(0)
{
    d_subKey = QString("dsa71604c");
    d_prettyName = QString("Ftmw Oscilloscope DSA71604C");

    d_comm = new TcpInstrument(d_key,d_subKey,this);
    connect(d_comm,&CommunicationProtocol::logMessage,this,&Dsa71604c::logMessage);
    connect(d_comm,&CommunicationProtocol::hardwareFailure,[=](){ emit hardwareFailure(this); });
    d_socket = dynamic_cast<TcpInstrument*>(d_comm)->d_socket;
}

Dsa71604c::~Dsa71604c()
{

}



bool Dsa71604c::testConnection()
{

    if(!d_comm->testConnection())
    {
        emit connected(false);
        return false;
    }

    QByteArray resp = scopeQueryCmd(QString("*IDN?\n"));

    if(resp.isEmpty())
    {
        emit connected(false,QString("Did not respond to ID query."));
        return false;
    }

    if(!resp.startsWith(QByteArray("TEKTRONIX,DSA71604C")))
    {
        emit connected(false,QString("ID response invalid. Response: %1 (Hex: %)").arg(QString(resp)).arg(QString(resp.toHex())));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp)));
    emit connected();
    return true;
}

void Dsa71604c::initialize()
{
    d_comm->setReadOptions(1000,true,QByteArray("\n"));
    d_comm->initialize();
    testConnection();
}

Experiment Dsa71604c::prepareForExperiment(Experiment exp)
{
    //attempt to apply settings. return invalid configuration if anything fails.
    //this is a lot of really tedious code.
    //All settings need to be made, and most of them also need to be verified
    //error messages are spit out to the UI
    //If this frequently fails, I recommend turning verbose headers on and writing a custom query command that verifies the header response, retrying until a valid reply is received.

    //make a copy of the configuration in which to store settings
    if(!exp.ftmwConfig().isEnabled())
        return exp;

    FtmwConfig::ScopeConfig config(exp.ftmwConfig().scopeConfig());
    disconnect(d_socket,&QTcpSocket::readyRead,this,&Dsa71604c::readWaveform);

    //disable ugly headers
    if(!d_comm->writeCmd(QString(":HEADER OFF\n")))
    {
        emit logMessage(QString("Could not disable verbose header mode."),LogHandler::Error);
        exp.setHardwareFailed();
        return exp;
    }

    //write data transfer commands
    if(!d_comm->writeCmd(QString(":DATA:SOURCE CH%1;START 1;STOP 1E12\n").arg(config.fidChannel)))
    {
        emit logMessage(QString("Could not write :DATA commands."),LogHandler::Error);
        exp.setHardwareFailed();
        return exp;
    }

    //clear out socket before senting our first query
    if(d_socket->bytesAvailable())
        d_socket->readAll();

    //verify that FID channel was set correctly
    QByteArray resp = scopeQueryCmd(QString(":DATA:SOURCE?\n"));
    if(resp.isEmpty() || !resp.contains(QString("CH%1").arg(config.fidChannel).toLatin1()))
    {
        emit logMessage(QString("Failed to set FID channel. Response to data source query: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
        exp.setHardwareFailed();
        return exp;
    }

//    if(!d_comm->writeCmd(QString("CH%1:BANDWIDTH:ENHANCED OFF; CH%1:BANDWIDTH 1.6+10; COUPLING AC;OFFSET 0;SCALE %2\n").arg(config.fidChannel).arg(QString::number(config.vScale,'g',4))))
    if(!d_comm->writeCmd(QString("CH%1:BANDWIDTH FULL; COUPLING AC;OFFSET 0;SCALE %2\n").arg(config.fidChannel).arg(QString::number(config.vScale,'g',4))))
    {
        emit logMessage(QString("Failed to write channel settings."),LogHandler::Error);
        exp.setHardwareFailed();
        return exp;
    }

    //read actual offset and vertical scale
    resp = scopeQueryCmd(QString(":CH%1:OFFSET?\n").arg(config.fidChannel));
    if(!resp.isEmpty())
    {
        bool ok = false;
        double offset = resp.trimmed().toDouble(&ok);
        if(!ok)
        {
            emit logMessage(QString("Could not parse offset response. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
        config.vOffset = offset;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to offset query."),LogHandler::Error);
        exp.setHardwareFailed();
        return exp;
    }
    resp = scopeQueryCmd(QString(":CH%1:SCALE?\n").arg(config.fidChannel));
    if(!resp.isEmpty())
    {
        bool ok = false;
        double scale = resp.trimmed().toDouble(&ok);
        if(!ok)
        {
            emit logMessage(QString("Could not parse scale response. Response: %2 (Hex: %3)").arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
        if(!(fabs(config.vScale-scale) < 0.01))
            emit logMessage(QString("Vertical scale is different than specified. Target: %1 V/div, Scope setting: %2 V/div").arg(QString::number(config.vScale,'f',3))
                            .arg(QString::number(scale,'f',3)),LogHandler::Warning);
        config.vScale = scale;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to scale query."),LogHandler::Error);
        exp.setHardwareFailed();
        return exp;
    }

    //horizontal settings
    if(!d_comm->writeCmd(QString(":HORIZONTAL:MODE MANUAL;POSITION 0;:HORIZONTAL:MODE:SAMPLERATE %1;RECORDLENGTH %2\n").arg(QString::number(config.sampleRate,'g',6)).arg(config.recordLength)))
    {
        emit logMessage(QString("Could not apply horizontal settings."),LogHandler::Error);
        exp.setHardwareFailed();
        return exp;
    }

    //verify sample rate and record length
    resp = scopeQueryCmd(QString(":HORIZONTAL:MODE:SAMPLERATE?\n"));
    if(!resp.isEmpty())
    {
        bool ok = false;
        double sRate = resp.trimmed().toDouble(&ok);
        if(!ok)
        {
            emit logMessage(QString("Sample rate query returned an invalid response. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
        if(!(fabs(sRate - config.sampleRate)<1e6))
        {
            emit logMessage(QString("Could not set sample rate successfully. Target: %1 GS/s, Scope setting: %2 GS/s").arg(QString::number(config.sampleRate/1e9,'f',3))
                            .arg(QString::number(sRate/1e9,'f',3)),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
        config.sampleRate = sRate;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to sample rate query."),LogHandler::Error);
        exp.setHardwareFailed();
        return exp;
    }
    resp = scopeQueryCmd(QString(":HORIZONTAL:MODE:RECORDLENGTH?\n"));
    if(!resp.isEmpty())
    {
        bool ok = false;
        int recLength = resp.trimmed().toInt(&ok);
        if(!ok)
        {
            emit logMessage(QString("Record length query returned an invalid response. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
        if(!(abs(recLength-config.recordLength) < 1000))
        {
            emit logMessage(QString("Could not set record length successfully! Target: %1, Scope setting: %2").arg(QString::number(config.recordLength))
                            .arg(QString::number(recLength)),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
        config.recordLength = recLength;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to record length query."),LogHandler::Error);
        exp.setHardwareFailed();
        return exp;
    }

    //fast frame settings
    if(!config.fastFrameEnabled)
    {
        resp = scopeQueryCmd(QString(":HORIZONTAL:FASTFRAME:STATE OFF;STATE?\n"));
        if(!resp.isEmpty())
        {
            bool ok = false;
            bool ffState = (bool)resp.trimmed().toInt(&ok);
            if(!ok || ffState)
            {
                emit logMessage(QString("Could not disable FastFrame mode."),LogHandler::Error);
                exp.setHardwareFailed();
                return exp;
            }
        }
        else
        {
            emit logMessage(QString("%1 gave an empty response to FastFrame state query."),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
    }
    else
    {
        //enable fastframe; verify
        resp = scopeQueryCmd(QString(":HORIZONTAL:FASTFRAME:STATE ON;STATE?\n"));
        if(!resp.isEmpty())
        {
            bool ok = false;
            bool ffState = (bool)resp.trimmed().toInt(&ok);
            if(!ok || !ffState)
            {
                emit logMessage(QString("Could not enable FastFrame mode."),LogHandler::Error);
                exp.setHardwareFailed();
                return exp;
            }
        }
        else
        {
            emit logMessage(QString("%1 gave an empty response to FastFrame state query."),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
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
                exp.setHardwareFailed();
                return exp;
            }

            //cap requested number of frames if it is greater than max
            int numFrames = config.numFrames;
            if(config.summaryFrame)
                numFrames++;

            if(maxFrames < numFrames)
            {
                emit logMessage(QString("Requested number of Fast frames (%1) is greater than maximum possible value with the requested acquisition settings (%2). Setting number of frames to %2.")
                                .arg(config.numFrames).arg(maxFrames),LogHandler::Warning);
                numFrames = maxFrames;
            }

            resp = scopeQueryCmd(QString(":HORIZONTAL:FASTFRAME:COUNT %1;COUNT?\n").arg(numFrames));
            if(!resp.isEmpty())
            {
                ok = false;
                int n = resp.trimmed().toInt(&ok);
                if(!ok)
                {
                    emit logMessage(QString("FastFrame count query returned an invalid response. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
                    exp.setHardwareFailed();
                    return exp;
                }
                if(n != numFrames)
                {
                    emit logMessage(QString("Requested number of FastFrames (%1) is different than actual number (%2). %2 frames will be acquired.").arg(numFrames).arg(n));
                }
                config.numFrames = n;
            }
            else
            {
                emit logMessage(QString("%1 gave an empty response to FastFrame count query."),LogHandler::Error);
                exp.setHardwareFailed();
                return exp;
            }

            QString sumfConfig = QString("AVE");
            if(!config.summaryFrame)
                sumfConfig = QString("NON");
            resp = scopeQueryCmd(QString(":HORIZONTAL:FASTFRAME:SUMFRAME %1;SUMFRAME?\n").arg(sumfConfig));
            if(!resp.isEmpty())
            {
                if(!QString(resp).contains(sumfConfig,Qt::CaseInsensitive))
                {
                    emit logMessage(QString("Could not configure FastFrame summary frame to %1. Response: %2 (Hex: %3)").arg(sumfConfig).arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
                    exp.setHardwareFailed();
                    return exp;
                }
            }
            else
            {
                emit logMessage(QString("%1 gave an empty response to FastFrame summary frame query."),LogHandler::Error);
                exp.setHardwareFailed();
                return exp;
            }
            if(config.summaryFrame)
            {
                //this forces the scope to only return the final frame, which is the summary frame
                if(!d_comm->writeCmd(QString(":DATA:FRAMESTART 100000;FRAMESTOP 100000\n")))
                {
                    emit logMessage(QString("Could not configure summary frame."),LogHandler::Error);
                    exp.setHardwareFailed();
                    return exp;
                }
            }
            else
            {
                //this forces the scope to return all frames
                if(!d_comm->writeCmd(QString(":DATA:FRAMESTART 1;FRAMESTOP 100000\n")))
                {
                    emit logMessage(QString("Could not configure frames."),LogHandler::Error);
                    exp.setHardwareFailed();
                    return exp;
                }
            }
        }
        else
        {
            emit logMessage(QString("%1 gave an empty response to FastFrame max frames query."),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
    }

    //trigger settings
    QString slope = QString("RIS");
    if(config.slope == FtmwConfig::FallingEdge)
        slope = QString("FALL");
    resp = scopeQueryCmd(QString(":TRIGGER:A:EDGE:SOURCE CH%2;COUPLING DC;SLOPE %1;:TRIGGER:A:LEVEL 0.35;:TRIGGER:A:EDGE:SOURCE?;SLOPE?\n").arg(slope).arg(config.trigChannel));
    if(!resp.isEmpty())
    {
        if(!QString(resp).contains(QString("CH%1").arg(config.trigChannel),Qt::CaseInsensitive))
        {
            emit logMessage(QString("Could not verify trigger channel. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }

        if(!QString(resp).contains(slope,Qt::CaseInsensitive))
        {
            emit logMessage(QString("Could not verify trigger slope. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
    }
    else
    {
        emit logMessage(QString("%1 gave an empty response to trigger query."),LogHandler::Error);
        exp.setHardwareFailed();
        return exp;
    }

    //set waveform output settings
    if(!d_comm->writeCmd(QString(":WFMOUTPRE:ENCDG BIN;BN_FMT RI;BYT_OR LSB;BYT_NR 1\n")))
    {
        emit logMessage(QString("Could not send waveform output commands."),LogHandler::Error);
        exp.setHardwareFailed();
        return exp;
    }

    //acquisition settings
    if(!d_comm->writeCmd(QString(":ACQUIRE:MODE SAMPLE;STOPAFTER RUNSTOP;STATE RUN\n")))
    {
        emit logMessage(QString("Could not send acquisition commands."),LogHandler::Error);
        exp.setHardwareFailed();
        return exp;
    }

    //force a trigger event to update these settings
    if(config.fastFrameEnabled)
    {
        for(int i=0;i<config.numFrames;i++)
            d_comm->writeCmd(QString(":TRIGGER FORCE\n"));
    }
    else
        d_comm->writeCmd(QString(":TRIGGER FORCE\n"));

    d_socket->waitForReadyRead(100);

    //read certain output settings from scope
    resp = scopeQueryCmd(QString(":WFMOUTPRE:ENCDG?;BN_FMT?;BYT_OR?;NR_FR?;NR_PT?;YMULT?;YOFF?;XINCR?;BYT_NR?\n"));
    if(!resp.isEmpty())
    {
        QStringList l = QString(resp.trimmed()).split(QChar(';'),QString::SkipEmptyParts);
        if(l.size() != 9)
        {
            emit logMessage(QString("Could not parse response to waveform output settings query. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }

        //check encoding
        if(!l.at(0).contains(QString("BIN"),Qt::CaseInsensitive))
        {
            emit logMessage(QString("Waveform encoding could not be set to binary. Response: %1 (Hex: %2)")
                            .arg(l.at(0)).arg(QString(l.at(0).toLatin1().toHex())),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
        //check binary format
        if(!l.at(1).contains(QString("RI"),Qt::CaseInsensitive))
        {
            emit logMessage(QString("Waveform format could not be set to signed integer. Response: %1 (Hex: %2)").arg(l.at(1)).arg(QString(l.at(1).toLatin1().toHex())),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
        //check byte order
        if(!l.at(2).contains(QString("LSB"),Qt::CaseInsensitive))
        {
            emit logMessage(QString("Waveform format could not be set to least significant byte first. Response: %1 (Hex: %2)")
                            .arg(l.at(2)).arg(QString(l.at(2).toLatin1().toHex())),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
        config.byteOrder = QDataStream::LittleEndian;

        //verify number of frames
        if(!config.summaryFrame && l.at(3).toInt() != config.numFrames)
        {
            emit logMessage(QString("Waveform contains the wrong number of frames. Target: %1, Actual: %2. Response: %3 (Hex: %4)")
                            .arg(config.numFrames).arg(l.at(3).toInt()).arg(l.at(3)).arg(QString(l.at(3).toLatin1().toHex())),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
        else if (config.summaryFrame && l.at(3).toInt() != 1)
        {
            emit logMessage(QString("Waveform contains the wrong number of frames. Target: 1 summary frame, Actual: %1. Response: %2 (Hex: %3)")
                            .arg(l.at(3).toInt()).arg(l.at(3)).arg(QString(l.at(3).toLatin1().toHex())),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
        //verify record length
        bool ok = false;
        int recLen = l.at(4).toInt(&ok);
        if(!ok)
        {
            emit logMessage(QString("Could not parse waveform record length response. Response: %1 (Hex: %2)")
                            .arg(l.at(4)).arg(QString(l.at(4).toLatin1().toHex())),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
        if(recLen != config.recordLength)
        {
            emit logMessage(QString("Record length is %1. Requested value was %2. Proceeding with %1 samples.")
                            .arg(recLen).arg(config.recordLength),LogHandler::Warning);
            config.recordLength = recLen;
        }
        //get y multiplier
        double ym = l.at(5).toDouble(&ok);
        if(!ok)
        {
            emit logMessage(QString("Could not parse waveform Y multiplier response. Response: %1 (Hex: %2)")
                            .arg(l.at(5)).arg(QString(l.at(5).toLatin1().toHex())),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
        config.yMult = ym;
        //get y offset
        double yo = l.at(6).toDouble(&ok);
        if(!ok)
        {
            emit logMessage(QString("Could not parse waveform Y offset response. Response: %1 (Hex: %2)")
                            .arg(l.at(6)).arg(QString(l.at(6).toLatin1().toHex())),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
        config.yOff = (int)round(yo);
        //get x increment
        double xi = l.at(7).toDouble(&ok);
        if(!ok)
        {
            emit logMessage(QString("Could not parse waveform X increment response. Response: %1 (Hex: %2)")
                            .arg(l.at(7)).arg(QString(l.at(7).toLatin1().toHex())),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
        config.xIncr = xi;
        //verify byte number
        int bpp = l.at(8).toInt(&ok);
        if(!ok || bpp < 1 || bpp > 2)
        {
            emit logMessage(QString("Invalid response to bytes per point query. Response: %1 (Hex: %2)")
                            .arg(l.at(8)).arg(QString(l.at(8).toLatin1().toHex())),LogHandler::Error);
            exp.setHardwareFailed();
            return exp;
        }
        config.bytesPerPoint = bpp;
    }

    if(!d_comm->writeCmd(QString("CH%1:BANDWIDTH:ENHANCED OFF; CH%1:BANDWIDTH 1.6E10\n").arg(config.fidChannel)))
//    if(!d_comm->writeCmd(QString("CH%1:BANDWIDTH FULL").arg(config.fidChannel)))
    {
        emit logMessage(QString("Failed to write channel settings."),LogHandler::Error);
        exp.setHardwareFailed();
        return exp;
    }

    d_configuration = config;
    exp.setScopeConfig(config);

    //lock scope, turn off waveform display, connect signal-slot stuff
    d_comm->writeCmd(QString(":LOCK ALL;:DISPLAY:WAVEFORM OFF\n"));
    if(d_socket->bytesAvailable())
        d_socket->readAll();



    return exp;
}

void Dsa71604c::beginAcquisition()
{
    if(d_socket->bytesAvailable())
        d_socket->readAll();

    d_comm->writeCmd(QString(":CURVESTREAM?\n"));
    d_waitingForReply = true;
    d_foundHeader = false;
    d_headerNumBytes = 0;
    d_waveformBytes = 0;
    d_lastTrigger = QDateTime::currentDateTime();
    connect(&d_scopeTimeout,&QTimer::timeout,this,&Dsa71604c::wakeUp,Qt::UniqueConnection);
    connect(d_socket,&QTcpSocket::readyRead,this,&Dsa71604c::readWaveform,Qt::UniqueConnection);
}

void Dsa71604c::endAcquisition()
{

    //stop parsing waveforms
    disconnect(d_socket,&QTcpSocket::readyRead,this,&Dsa71604c::readWaveform);
    disconnect(&d_scopeTimeout,&QTimer::timeout,this,&Dsa71604c::wakeUp);

    if(d_socket->bytesAvailable())
        d_socket->readAll();

    //send *CLS command twice to kick scope out of curvestream mode and clear the error queue
    d_comm->writeCmd(QString("*CLS\n"));
    d_comm->writeCmd(QString("*CLS\n"));

    if(d_socket->bytesAvailable())
        d_socket->readAll();

    d_comm->writeCmd(QString(":UNLOCK ALL;:DISPLAY:WAVEFORM ON\n"));

    d_waitingForReply = false;
    d_foundHeader = false;
    d_headerNumBytes = 0;
    d_waveformBytes = 0;
}

void Dsa71604c::readTimeData()
{
}

void Dsa71604c::readWaveform()
{
    if(!d_waitingForReply) // if for some reason the readyread signal weren't disconnected, don't eat all the bytes
        return;

    qint64 ba = d_socket->bytesAvailable();
    emit logMessage(QString("Bytes available: %1\t%2 ms").arg(ba).arg(QTime::currentTime().msec()));

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
            d_socket->getChar(&c);
            if(c=='#')
            {
                d_foundHeader = true;
                emit logMessage(QString("Found hdr: %1 ms").arg(QTime::currentTime().msec()));
            }
            i++;
        }
    }

    if(d_foundHeader && d_headerNumBytes == 0) //we've found the header hash, now get the number of header bytes
    {
        //make sure the entire header can be read
        if(d_socket->bytesAvailable())
        {
            //there is a header byte ready. Read it and set the number of remaining header bytes
            char c=0;
            d_socket->getChar(&c);
            QString hdrNum = QString::fromLatin1(&c,1);
            bool ok = false;
            int nb = hdrNum.toInt(&ok,16);
            if(!ok || nb < 1 || nb > 15)
            {
                //it's possible that we're in the middle of an old waveform by fluke.
                //continue looking for '#'
                d_foundHeader = false;
                if(d_socket->bytesAvailable())
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
        if(d_socket->bytesAvailable() >= d_headerNumBytes)
        {
            QByteArray wfmBytes = d_socket->read(d_headerNumBytes);
            bool ok = false;
            int b = wfmBytes.toInt(&ok);
            int nf = d_configuration.numFrames;
            if(d_configuration.summaryFrame || !d_configuration.fastFrameEnabled)
                nf = 1;
            if(!ok || b != d_configuration.recordLength*d_configuration.bytesPerPoint*nf)
            {
                //it's possible that we're in the middle of an old waveform by fluke.
                //continue looking for '#'
                d_foundHeader = false;
                d_headerNumBytes = 0;
                if(d_socket->bytesAvailable())
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
        if(d_socket->bytesAvailable() >= d_waveformBytes) // whole waveform can be read!
        {
            QByteArray wfm = d_socket->read(d_waveformBytes);
            emit logMessage(QString("Wfm read complete: %1 ms").arg(QTime::currentTime().msec()));
            emit shotAcquired(wfm);
            d_waitingForReply = false;
            d_foundHeader = false;
            d_headerNumBytes = 0;
            d_waveformBytes = 0;
            return;
        }
        else
            return;
    }
}

void Dsa71604c::queryScope()
{
    emit logMessage(QString("Trigger detected: %1 ms").arg(QTime::currentTime().msec()));

    d_scopeTimeout.stop();
//    d_scopeTimeout.start(10000);

    if(d_waitingForReply) // a previous query has been sent, maybe the transfer isn't done
    {
        if(d_foundHeader && d_lastTrigger.addSecs(10) < QDateTime::currentDateTime()) // previous acquisition is just waiting for transfer to finish, so don't retrigger
            return;

//        if(d_lastTrigger.addMSecs(1000) < QDateTime::currentDateTime()) //we've gone 5 seconds without a trigger. Need to wake up!
//        {
//            d_lastTrigger = QDateTime::currentDateTime();
//            wakeUp();
//            return;
//        }

//        return;
    }

//    if(d_socket->bytesAvailable())
//        d_socket->readAll();

    d_waitingForReply = true;
    d_foundHeader = false;
    d_headerNumBytes = 0;
    d_waveformBytes = 0;
    //the scope appears to cache replies, so each waveform is sent on the following trigger event
    d_lastTrigger = QDateTime::currentDateTime();

    //    d_comm->writeCmd(QString(":CURVE?\n"));
}

void Dsa71604c::wakeUp()
{
    d_scopeTimeout.stop();
    emit logMessage(QString("Attempting to wake up scope"),LogHandler::Warning);

    endAcquisition();

    d_socket->waitForReadyRead();

//    if(!testConnection())
//        return;

    beginAcquisition();
}

QByteArray Dsa71604c::scopeQueryCmd(QString query)
{
    //the scope is flaky. Sometimes, for no apparent reason, it doesn't respond
    //This will retry the query if it fails, suppressing any errors on the first try

    blockSignals(true);
    QByteArray resp = d_comm->queryCmd(query);
    blockSignals(false);

    if(resp.isEmpty())
        resp = d_comm->queryCmd(query);

    return resp;
}
