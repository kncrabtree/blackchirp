#include "mso72004c.h"

#include <QTcpSocket>
#include <math.h>

MSO72004C::MSO72004C(QObject *parent) :
    FtmwScope(parent), d_waitingForReply(false), d_foundHeader(false),
    d_headerNumBytes(0), d_waveformBytes(0)
{
    d_subKey = QString("MSO72004C");
    d_prettyName = QString("Ftmw Oscilloscope MSO72004C");
    d_commType = CommunicationProtocol::Tcp;

}

MSO72004C::~MSO72004C()
{

}

void MSO72004C::readSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    s.setValue(QString("canBlockAverage"),false);
    s.setValue(QString("canFastFrame"),true);
    s.setValue(QString("canSummaryFrame"),true);
    s.setValue(QString("canBlockAndFastFrame"),false);

    double bandwidth = s.value(QString("bandwidth"),16000.0).toDouble();
    s.setValue(QString("bandwidth"),bandwidth);

    if(s.beginReadArray(QString("sampleRates")) < 1)
    {
        s.endArray();
        QList<QPair<QString,double>> sampleRates;
        sampleRates << qMakePair(QString("2 GSa/s"),2e9) << qMakePair(QString("5 GSa/s"),5e9)  << qMakePair(QString("10 GSa/s"),10e9)
                    << qMakePair(QString("20 GSa/s"),20e9) << qMakePair(QString("50 GSa/s"),50e9)  << qMakePair(QString("100 GSa/s"),100e9);

        s.beginWriteArray(QString("sampleRates"));
        for(int i=0; i<sampleRates.size(); i++)
        {
            s.setArrayIndex(i);
            s.setValue(QString("text"),sampleRates.at(i).first);
            s.setValue(QString("val"),sampleRates.at(i).second);
        }
    }
    s.endArray();
    s.endGroup();
    s.endGroup();
}



bool MSO72004C::testConnection()
{
    QByteArray resp = scopeQueryCmd(QString("*IDN?\n"));

    if(resp.isEmpty())
    {
        d_errorString = QString("Did not respond to ID query.");
        return false;
    }

    if(resp.length() > 100)
        resp = resp.mid(0,100);

    if(!resp.startsWith(QByteArray("TEKTRONIX,MSO")))
    {
        d_errorString = QString("ID response invalid. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex()));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp)));
    return true;
}

void MSO72004C::initialize()
{
    p_scopeTimeout = new QTimer(this);

    p_comm->setReadOptions(1000,true,QByteArray("\n"));
    p_socket = dynamic_cast<QTcpSocket*>(p_comm->device());
    connect(p_socket,static_cast<void (QTcpSocket::*)(QAbstractSocket::SocketError)>(&QTcpSocket::error),this,&MSO72004C::socketError);
    p_socket->setSocketOption(QAbstractSocket::LowDelayOption,1);
}

bool MSO72004C::prepareForExperiment(Experiment &exp)
{
    //attempt to apply settings. return invalid configuration if anything fails.
    //this is a lot of really tedious code.
    //All settings need to be made, and most of them also need to be verified
    //error messages are spit out to the UI
    //If this frequently fails, I recommend turning verbose headers on and writing a custom query command that verifies the header response, retrying until a valid reply is received.

    //make a copy of the configuration in which to store settings
    d_enabledForExperiment = exp.ftmwConfig().isEnabled();
    if(!d_enabledForExperiment)
        return true;

    BlackChirp::FtmwScopeConfig config(exp.ftmwConfig().scopeConfig());
    disconnect(p_socket,&QTcpSocket::readyRead,this,&MSO72004C::readWaveform);

    //disable ugly headers
    if(!p_comm->writeCmd(QString(":HEADER OFF\n")))
    {
        emit logMessage(QString("Could not disable verbose header mode."),BlackChirp::LogError);
        exp.setHardwareFailed();
        return false;
    }

    //write data transfer commands
    if(!p_comm->writeCmd(QString(":DATA:SOURCE CH%1;START 1;STOP 1E12\n").arg(config.fidChannel)))
    {
        emit logMessage(QString("Could not write :DATA commands."),BlackChirp::LogError);
        exp.setHardwareFailed();
        return false;
    }

    //clear out socket before senting our first query
    if(p_socket->bytesAvailable())
        p_socket->readAll();

    //verify that FID channel was set correctly
    QByteArray resp = scopeQueryCmd(QString(":DATA:SOURCE?\n"));
    if(resp.isEmpty() || !resp.contains(QString("CH%1").arg(config.fidChannel).toLatin1()))
    {
        emit logMessage(QString("Failed to set FID channel. Response to data source query: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),BlackChirp::LogError);
        exp.setHardwareFailed();
        return false;
    }

    if(!p_comm->writeCmd(QString(":CH%1:BANDWIDTH:ENHANCED OFF; :CH%1:BANDWIDTH 1.6E+10; COUPLING AC;POSITION 0;OFFSET 0;SCALE %2\n")
                         .arg(config.fidChannel).arg(QString::number(config.vScale,'e',3))))
    {
        emit logMessage(QString("Failed to write channel settings."),BlackChirp::LogError);
        exp.setHardwareFailed();
        return false;
    }

    //read actual offset and vertical scale
    resp = scopeQueryCmd(QString(":CH%1:OFFSET?\n").arg(config.fidChannel));
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
    resp = scopeQueryCmd(QString(":CH%1:SCALE?\n").arg(config.fidChannel));
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

    //horizontal settings
    if(!p_comm->writeCmd(QString(":HORIZONTAL:MODE MANUAL;:HORIZONTAL:DELAY:MODE ON;:HORIZONTAL:DELAY:POSITION 0;:HORIZONTAL:DELAY:TIME %1;:HORIZONTAL:MODE:SAMPLERATE %2;RECORDLENGTH %3\n")
                         .arg(QString::number(config.trigDelay,'g',6)).arg(QString::number(config.sampleRate,'g',6)).arg(config.recordLength)))
    {
        emit logMessage(QString("Could not apply horizontal settings."),BlackChirp::LogError);
        exp.setHardwareFailed();
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
    resp = scopeQueryCmd(QString(":HORIZONTAL:MODE:RECORDLENGTH?\n"));
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
            emit logMessage(QString("Could not set record length successfully! Target: %1, Scope setting: %2").arg(QString::number(config.recordLength))
                            .arg(QString::number(recLength)),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
        config.recordLength = recLength;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to record length query."),BlackChirp::LogError);
        exp.setHardwareFailed();
        return false;
    }
    resp = scopeQueryCmd(QString(":HORIZONTAL:DELAY:TIME?\n"));
    if(!resp.isEmpty())
    {
        bool ok = false;
        double delay = resp.trimmed().toDouble(&ok);
        if(!ok)
        {
            emit logMessage(QString("Trigger delay query returned an invalid response. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
        if(!qFuzzyCompare(1.0+delay,1.0+config.trigDelay))
        {
            emit logMessage(QString("Could not set trigger delay successfully! Target: %1, Scope setting: %2").arg(QString::number(config.trigDelay))
                            .arg(QString::number(delay)),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
        config.trigDelay = delay;
    }
    else
    {
        emit logMessage(QString("Gave an empty response to trigger delay query."),BlackChirp::LogError);
        exp.setHardwareFailed();
        return false;
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
                emit logMessage(QString("Could not disable FastFrame mode."),BlackChirp::LogError);
                exp.setHardwareFailed();
                return false;
            }
        }
        else
        {
            emit logMessage(QString("Gave an empty response to FastFrame state query."),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
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
                emit logMessage(QString("Could not enable FastFrame mode."),BlackChirp::LogError);
                exp.setHardwareFailed();
                return false;
            }
        }
        else
        {
            emit logMessage(QString("Gave an empty response to FastFrame state query."),BlackChirp::LogError);
            exp.setHardwareFailed();
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
                emit logMessage(QString("Could not determine maximum number of frames in FastFrame mode."),BlackChirp::LogError);
                exp.setHardwareFailed();
                return false;
            }

            //cap requested number of frames if it is greater than max
            int numFrames = config.numFrames;
            if(config.summaryFrame)
                numFrames++;

            if(maxFrames < numFrames)
            {
                emit logMessage(QString("Requested number of Fast frames (%1) is greater than maximum possible value with the requested acquisition settings (%2). Setting number of frames to %2.")
                                .arg(config.numFrames).arg(maxFrames),BlackChirp::LogWarning);
                numFrames = maxFrames;
            }

            resp = scopeQueryCmd(QString(":HORIZONTAL:FASTFRAME:COUNT %1;COUNT?\n").arg(numFrames));
            if(!resp.isEmpty())
            {
                ok = false;
                int n = resp.trimmed().toInt(&ok);
                if(!ok)
                {
                    emit logMessage(QString("FastFrame count query returned an invalid response. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),BlackChirp::LogError);
                    exp.setHardwareFailed();
                    return false;
                }
                if(n != numFrames)
                {
                    emit logMessage(QString("Requested number of FastFrames (%1) is different than actual number (%2). %2 frames will be acquired.").arg(numFrames).arg(n));
                }
                if(config.summaryFrame)
                    config.numFrames = n-1;
                else
                    config.numFrames = n;
            }
            else
            {
                emit logMessage(QString("Gave an empty response to FastFrame count query."),BlackChirp::LogError);
                exp.setHardwareFailed();
                return false;
            }

            QString sumfConfig = QString("AVE");
            if(!config.summaryFrame)
                sumfConfig = QString("NON");
            resp = scopeQueryCmd(QString(":HORIZONTAL:FASTFRAME:SUMFRAME %1;SUMFRAME?\n").arg(sumfConfig));
            if(!resp.isEmpty())
            {
                if(!QString(resp).contains(sumfConfig,Qt::CaseInsensitive))
                {
                    emit logMessage(QString("Could not configure FastFrame summary frame to %1. Response: %2 (Hex: %3)").arg(sumfConfig).arg(QString(resp)).arg(QString(resp.toHex())),BlackChirp::LogError);
                    exp.setHardwareFailed();
                    return false;
                }
            }
            else
            {
                emit logMessage(QString("Gave an empty response to FastFrame summary frame query."),BlackChirp::LogError);
                exp.setHardwareFailed();
                return false;
            }
            if(config.summaryFrame)
            {
                //this forces the scope to only return the final frame, which is the summary frame
                if(!p_comm->writeCmd(QString(":DATA:FRAMESTART 100000;FRAMESTOP 100000\n")))
                {
                    emit logMessage(QString("Could not configure summary frame."),BlackChirp::LogError);
                    exp.setHardwareFailed();
                    return false;
                }
            }
            else
            {
                //this forces the scope to return all frames
                if(!p_comm->writeCmd(QString(":DATA:FRAMESTART 1;FRAMESTOP 100000\n")))
                {
                    emit logMessage(QString("Could not configure frames."),BlackChirp::LogError);
                    exp.setHardwareFailed();
                    return false;
                }
            }
        }
        else
        {
            emit logMessage(QString("Gave an empty response to FastFrame max frames query."),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
    }

    //trigger settings
    QString slope = QString("RIS");
    if(config.slope == BlackChirp::FallingEdge)
        slope = QString("FALL");
    QString trigCh = QString("AUX");
    if(config.trigChannel > 0)
        trigCh = QString("CH%1").arg(config.trigChannel);

    resp = scopeQueryCmd(QString(":TRIGGER:A:EDGE:SOURCE %1;COUPLING DC;SLOPE %2;:TRIGGER:A:LEVEL %3;:TRIGGER:A:EDGE:SOURCE?;SLOPE?\n").arg(trigCh).arg(slope).arg(config.trigLevel,0,'f',3));
    if(!resp.isEmpty())
    {
        if(!QString(resp).contains(trigCh),Qt::CaseInsensitive)
        {
            emit logMessage(QString("Could not verify trigger channel. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }

        if(!QString(resp).contains(slope,Qt::CaseInsensitive))
        {
            emit logMessage(QString("Could not verify trigger slope. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
    }
    else
    {
        emit logMessage(QString("Gave an empty response to trigger query."),BlackChirp::LogError);
        exp.setHardwareFailed();
        return false;
    }

    //set waveform output settings
    if(!p_comm->writeCmd(QString(":WFMOUTPRE:ENCDG BIN;BN_FMT RI;BYT_OR LSB;BYT_NR %1\n").arg(config.bytesPerPoint)))
    {
        emit logMessage(QString("Could not send waveform output commands."),BlackChirp::LogError);
        exp.setHardwareFailed();
        return false;
    }

    //acquisition settings
    if(!p_comm->writeCmd(QString(":ACQUIRE:MODE SAMPLE;STOPAFTER RUNSTOP;STATE RUN\n")))
    {
        emit logMessage(QString("Could not send acquisition commands."),BlackChirp::LogError);
        exp.setHardwareFailed();
        return false;
    }

    //force a trigger event to update these settings
    if(config.fastFrameEnabled)
    {
        for(int i=0;i<config.numFrames;i++)
            p_comm->writeCmd(QString(":TRIGGER FORCE\n"));
    }
    else
        p_comm->writeCmd(QString(":TRIGGER FORCE\n"));

    //read certain output settings from scope
    resp = scopeQueryCmd(QString(":WFMOUTPRE:ENCDG?;BN_FMT?;BYT_OR?;NR_FR?;NR_PT?;YMULT?;YOFF?;XINCR?;BYT_NR?\n"));
    if(!resp.isEmpty())
    {
        QStringList l = QString(resp.trimmed()).split(QChar(';'),QString::SkipEmptyParts);
        if(l.size() != 9)
        {
            emit logMessage(QString("Could not parse response to waveform output settings query. Response: %1 (Hex: %2)").arg(QString(resp)).arg(QString(resp.toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }

        //check encoding
        if(!l.at(0).contains(QString("BIN"),Qt::CaseInsensitive))
        {
            emit logMessage(QString("Waveform encoding could not be set to binary. Response: %1 (Hex: %2)")
                            .arg(l.at(0)).arg(QString(l.at(0).toLatin1().toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
        //check binary format
        if(!l.at(1).contains(QString("RI"),Qt::CaseInsensitive))
        {
            emit logMessage(QString("Waveform format could not be set to signed integer. Response: %1 (Hex: %2)").arg(l.at(1)).arg(QString(l.at(1).toLatin1().toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
        //check byte order
        if(!l.at(2).contains(QString("LSB"),Qt::CaseInsensitive))
        {
            emit logMessage(QString("Waveform format could not be set to least significant byte first. Response: %1 (Hex: %2)")
                            .arg(l.at(2)).arg(QString(l.at(2).toLatin1().toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
        config.byteOrder = QDataStream::LittleEndian;

        //verify number of frames
        if(config.fastFrameEnabled && !config.summaryFrame && l.at(3).toInt() != config.numFrames)
        {
            emit logMessage(QString("Waveform contains the wrong number of frames. Target: %1, Actual: %2. Response: %3 (Hex: %4)")
                            .arg(config.numFrames).arg(l.at(3).toInt()).arg(l.at(3)).arg(QString(l.at(3).toLatin1().toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
        else if (config.summaryFrame && l.at(3).toInt() != 1)
        {
            emit logMessage(QString("Waveform contains the wrong number of frames. Target: 1 summary frame, Actual: %1. Response: %2 (Hex: %3)")
                            .arg(l.at(3).toInt()).arg(l.at(3)).arg(QString(l.at(3).toLatin1().toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
        //verify record length
        bool ok = false;
        int recLen = l.at(4).toInt(&ok);
        if(!ok)
        {
            emit logMessage(QString("Could not parse waveform record length response. Response: %1 (Hex: %2)")
                            .arg(l.at(4)).arg(QString(l.at(4).toLatin1().toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
        if(recLen != config.recordLength)
        {
            emit logMessage(QString("Record length is %1. Requested value was %2. Proceeding with %1 samples.")
                            .arg(recLen).arg(config.recordLength),BlackChirp::LogWarning);
            config.recordLength = recLen;
        }
        //get y multiplier
        double ym = l.at(5).toDouble(&ok);
        if(!ok)
        {
            emit logMessage(QString("Could not parse waveform Y multiplier response. Response: %1 (Hex: %2)")
                            .arg(l.at(5)).arg(QString(l.at(5).toLatin1().toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
        config.yMult = ym;
        //get y offset
        double yo = l.at(6).toDouble(&ok);
        if(!ok)
        {
            emit logMessage(QString("Could not parse waveform Y offset response. Response: %1 (Hex: %2)")
                            .arg(l.at(6)).arg(QString(l.at(6).toLatin1().toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
        config.yOff = (int)round(yo);
        //get x increment
        double xi = l.at(7).toDouble(&ok);
        if(!ok)
        {
            emit logMessage(QString("Could not parse waveform X increment response. Response: %1 (Hex: %2)")
                            .arg(l.at(7)).arg(QString(l.at(7).toLatin1().toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
        config.xIncr = xi;
        //verify byte number
        int bpp = l.at(8).toInt(&ok);
        if(!ok || bpp < 1 || bpp > 2)
        {
            emit logMessage(QString("Invalid response to bytes per point query. Response: %1 (Hex: %2)")
                            .arg(l.at(8)).arg(QString(l.at(8).toLatin1().toHex())),BlackChirp::LogError);
            exp.setHardwareFailed();
            return false;
        }
        config.bytesPerPoint = bpp;
    }

    d_configuration = config;
    exp.setScopeConfig(config);

    if(p_socket->bytesAvailable())
        p_socket->readAll();



    return true;
}

void MSO72004C::beginAcquisition()
{
    if(d_enabledForExperiment)
    {
        p_comm->writeCmd(QString(":LOCK ALL;:DISPLAY:WAVEFORM OFF\n"));
        if(p_socket->bytesAvailable())
            p_socket->readAll();

        p_comm->writeCmd(QString(":CURVESTREAM?\n"));
        d_waitingForReply = true;
        d_foundHeader = false;
        d_headerNumBytes = 0;
        d_waveformBytes = 0;

        p_scopeTimeout->stop();
        p_scopeTimeout->start(10000);

        connect(p_scopeTimeout,&QTimer::timeout,this,&MSO72004C::wakeUp,Qt::UniqueConnection);
        connect(p_socket,&QTcpSocket::readyRead,this,&MSO72004C::readWaveform,Qt::UniqueConnection);
    }
}

void MSO72004C::endAcquisition()
{

    if(d_enabledForExperiment)
    {
        //stop parsing waveforms
        p_scopeTimeout->stop();
        disconnect(p_socket,&QTcpSocket::readyRead,this,&MSO72004C::readWaveform);
        disconnect(p_scopeTimeout,&QTimer::timeout,this,&MSO72004C::wakeUp);

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

void MSO72004C::readWaveform()
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
                p_scopeTimeout->start(10000);
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
            int nf = d_configuration.numFrames;
            if(d_configuration.summaryFrame || !d_configuration.fastFrameEnabled)
                nf = 1;
            if(!ok || b != d_configuration.recordLength*d_configuration.bytesPerPoint*nf)
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

void MSO72004C::wakeUp()
{
    p_scopeTimeout->stop();
    emit logMessage(QString("Attempting to wake up scope"),BlackChirp::LogWarning);

    endAcquisition();

    if(!testConnection())
    {
        emit hardwareFailure();
        return;
    }

    p_comm->writeCmd(QString(":LOCK ALL;:DISPLAY:WAVEFORM OFF\n"));
    beginAcquisition();
}

void MSO72004C::socketError(QAbstractSocket::SocketError e)
{
    emit logMessage(QString("Socket error: %1").arg((int)e),BlackChirp::LogError);
    emit logMessage(QString("Error message: %1").arg(p_socket->errorString()),BlackChirp::LogError);
    emit hardwareFailure();
}

QByteArray MSO72004C::scopeQueryCmd(QString query)
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
