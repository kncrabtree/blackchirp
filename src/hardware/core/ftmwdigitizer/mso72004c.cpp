#include "mso72004c.h"

#include <QTcpSocket>
#include <QThread>
#include <math.h>
#include <hardware/core/hardwareregistration.h>

using namespace BC::Key::FtmwDigitizer;
using namespace BC::Key::Digi;

// Register this hardware implementation
REGISTER_HARDWARE_META(MSO72004C, "Tektronix MSO72004C FTMW Digitizer (20 GHz, 50 GS/s)")
REGISTER_HARDWARE_PROTOCOLS(MSO72004C, CommunicationProtocol::Tcp)
REGISTER_HARDWARE_SETTINGS(MSO72004C,
    {maxRecordLength,    "Max Record Length",   "Maximum record length in samples",
     100000000, 0, QVariant{}, HwSettingPriority::Important}
)
REGISTER_HARDWARE_ARRAY(MSO72004C, sampleRates,
    "Sample Rates", "Available digitizer sample rates",
    HwSettingPriority::Important)
REGISTER_HARDWARE_ARRAY_ENTRY(MSO72004C, sampleRates,
    {{srText, "2 GSa/s"}, {srValue, 2e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(MSO72004C, sampleRates,
    {{srText, "5 GSa/s"}, {srValue, 5e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(MSO72004C, sampleRates,
    {{srText, "10 GSa/s"}, {srValue, 10e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(MSO72004C, sampleRates,
    {{srText, "20 GSa/s"}, {srValue, 20e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(MSO72004C, sampleRates,
    {{srText, "50 GSa/s"}, {srValue, 50e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(MSO72004C, sampleRates,
    {{srText, "100 GSa/s"}, {srValue, 100e9}})

MSO72004C::MSO72004C(const QString& label, QObject *parent) :
    FtmwDigitizer(QString(MSO72004C::staticMetaObject.className()), label, parent),
    d_waitingForReply(false), d_foundHeader(false),
    d_headerNumBytes(0), d_waveformBytes(0)
{

    // Communication defaults
    setDefault(BC::Key::Comm::timeout, 1000);
    setDefault(BC::Key::Comm::termChar, QString("\n"));

    save();
}

MSO72004C::~MSO72004C()
{

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

    hwDebug(u"%1: ID response: %2"_s.arg(d_key, QString(resp)));
    return true;
}

void MSO72004C::initialize()
{
    p_scopeTimeout = new QTimer(this);

    p_socket = p_comm->device<QTcpSocket>();
    connect(p_socket,static_cast<void (QTcpSocket::*)(QAbstractSocket::SocketError)>(&QTcpSocket::errorOccurred),this,&MSO72004C::socketError);
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
    d_enabledForExperiment = exp.ftmwEnabled();
    if(!d_enabledForExperiment)
        return true;

    auto &config = exp.ftmwConfig()->digitizerConfig();

    disconnect(p_socket,&QTcpSocket::readyRead,this,&MSO72004C::readWaveform);

    //disable ugly headers
    if(!p_comm->writeCmd(QString(":HEADER OFF\n")))
    {
        hwError("Could not disable verbose header mode."_L1);
        return false;
    }

    //write data transfer commands
    if(!p_comm->writeCmd(QString(":DATA:SOURCE CH%1;START 1;STOP 1E12\n").arg(config.d_fidChannel)))
    {
        hwError("Could not write :DATA commands."_L1);
        return false;
    }

    //clear out socket before senting our first query
    if(p_socket->bytesAvailable())
        p_socket->readAll();

    //verify that FID channel was set correctly
    QByteArray resp = scopeQueryCmd(QString(":DATA:SOURCE?\n"));
    if(resp.isEmpty() || !resp.contains(QString("CH%1").arg(config.d_fidChannel).toLatin1()))
    {
        hwError("Failed to set FID channel."_L1);
        hwDebug(u"%1: Failed to set FID channel. Response to data source query = %2 (Hex: %3)"_s
                    .arg(d_key, QString(resp), QString(resp.toHex())));
        return false;
    }

    if(!p_comm->writeCmd(QString("CH%1:BANDWIDTH:ENHANCED OFF; CH%1:BANDWIDTH 1.6+10; COUPLING AC;OFFSET %2;SCALE %3\n")
                         .arg(config.d_fidChannel)
                         .arg(QString::number(config.d_analogChannels[config.d_fidChannel].offset,'g',4))
                         .arg(QString::number(config.d_analogChannels[config.d_fidChannel].fullScale/5.0,'g',4))))
    {
        hwError("Failed to write channel settings."_L1);
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
            hwError("Could not parse offset response."_L1);
            hwDebug(u"%1: Could not parse offset response. Response = %2 (Hex: %3)"_s
                        .arg(d_key, QString(resp), QString(resp.toHex())));

            return false;
        }
        config.d_analogChannels[config.d_fidChannel].offset = offset;
    }
    else
    {
        hwError("Gave an empty response to offset query."_L1);
        return false;
    }

    resp = scopeQueryCmd(QString(":CH%1:SCALE?\n").arg(config.d_fidChannel));
    if(!resp.isEmpty())
    {
        bool ok = false;
        double scale = resp.trimmed().toDouble(&ok);
        if(!ok)
        {
            hwError("Could not parse scale response."_L1);
            hwDebug(u"%1: Could not parse scale response. Response = %2 (Hex: %3)"_s
                        .arg(d_key, QString(resp), QString(resp.toHex())));
            return false;
        }
        if(!(fabs(config.d_analogChannels[config.d_fidChannel].fullScale/5.0-scale) < 0.01))
            hwWarn(u"Vertical full scale is different than specified. Target: %1 V, Scope setting: %2 V"_s
                       .arg(QString::number(config.d_analogChannels[config.d_fidChannel].fullScale/5.0,'f',3),
                            QString::number(scale*5.0,'f',3)));
        config.d_analogChannels[config.d_fidChannel].fullScale = scale*5.0;
    }
    else
    {
        hwError("Gave an empty response to scale query."_L1);
        return false;
    }

    //horizontal settings
    if(!p_comm->writeCmd(QString(":HORIZONTAL:MODE MANUAL;:HORIZONTAL:DELAY:MODE ON;:HORIZONTAL:DELAY:POSITION 0;:HORIZONTAL:DELAY:TIME %1;:HORIZONTAL:MODE:SAMPLERATE %2;RECORDLENGTH %3\n")
                         .arg(QString::number(config.d_triggerDelayUSec,'g',6)).arg(QString::number(config.d_sampleRate,'g',6)).arg(config.d_recordLength)))
    {
        hwError("Could not apply horizontal settings."_L1);
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
            hwError("Sample rate query returned an invalid response."_L1);
            hwDebug(u"%1: Sample rate query returned an invalid response. Response = %2 (Hex: %3)"_s
                        .arg(d_key, QString(resp), QString(resp.toHex())));
            return false;
        }
        if(!(fabs(sRate - config.d_sampleRate)<1e6))
        {
            hwError(u"Could not set sample rate successfully. Target: %1 GS/s, Scope setting: %2 GS/s"_s
                        .arg(QString::number(config.d_sampleRate/1e9,'f',3),
                             QString::number(sRate/1e9,'f',3)));
            return false;
        }
        config.d_sampleRate = sRate;
    }
    else
    {
        hwError("Gave an empty response to sample rate query."_L1);
        return false;
    }
    resp = scopeQueryCmd(QString(":HORIZONTAL:MODE:RECORDLENGTH?\n"));
    if(!resp.isEmpty())
    {
        bool ok = false;
        int recLength = resp.trimmed().toInt(&ok);
        if(!ok)
        {
            hwError("Record length query returned an invalid response."_L1);
            hwDebug(u"%1: Record length query returned an invalid response. Response = %2 (Hex: %3)"_s
                        .arg(d_key, QString(resp), QString(resp.toHex())));
            return false;
        }
        if(!(abs(recLength-config.d_recordLength) < 1000))
        {
            hwError(u"Could not set record length successfully! Target: %1, Scope setting: %2"_s
                        .arg(config.d_recordLength).arg(recLength));
            return false;
        }
        config.d_recordLength = recLength;
    }
    else
    {
        hwError("Gave an empty response to record length query."_L1);
        return false;
    }
    resp = scopeQueryCmd(QString(":HORIZONTAL:DELAY:TIME?\n"));
    if(!resp.isEmpty())
    {
        bool ok = false;
        double delay = resp.trimmed().toDouble(&ok);
        if(!ok)
        {
            hwError("Trigger delay query returned an invalid response."_L1);
            hwDebug(u"%1: Trigger delay query returned an invalid response. Response = %2 (Hex: %3)"_s
                        .arg(d_key, QString(resp), QString(resp.toHex())));
            return false;
        }
        if(fabs(delay-config.d_triggerDelayUSec) > 1e-6)
        {
            hwError(u"Could not set trigger delay successfully! Target: %1, Scope setting: %2"_s
                        .arg(QString::number(config.d_triggerDelayUSec),
                             QString::number(delay)));
            return false;
        }
        config.d_triggerDelayUSec = delay;
    }
    else
    {
        hwError("Gave an empty response to trigger delay query."_L1);
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
                hwError("Could not disable FastFrame mode."_L1);
                return false;
            }
        }
        else
        {
            hwError("Gave an empty response to FastFrame state query."_L1);
            return false;
        }
    }
    else
    {
        //enable fastframe and disable summary frame; verify
        resp = scopeQueryCmd(QString(":HORIZONTAL:FASTFRAME:STATE ON;SUMFRAME NONE;STATE?\n"));
        if(!resp.isEmpty())
        {
            bool ok = false;
            bool ffState = (bool)resp.trimmed().toInt(&ok);
            if(!ok || !ffState)
            {
                hwError("Could not enable FastFrame mode."_L1);
                return false;
            }
        }
        else
        {
            hwError("Gave an empty response to FastFrame state query."_L1);
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
                hwError("Could not determine maximum number of frames in FastFrame mode."_L1);
                return false;
            }

            int numFrames = qMax(config.d_numRecords,config.d_numAverages);
            if(config.d_blockAverage)
                maxFrames -= 2; //note: if summary frame enabled, need to reserve 2 frames worth of memory

            if(maxFrames < numFrames)
            {
                hwError(u"Requested number of Fast frames (%1) is greater than maximum possible value with the requested acquisition settings (%2)."_s
                            .arg(numFrames).arg(maxFrames));
                return false;
            }

            resp = scopeQueryCmd(QString(":HORIZONTAL:FASTFRAME:COUNT %1;COUNT?\n").arg(numFrames));
            if(!resp.isEmpty())
            {
                ok = false;
                int n = resp.trimmed().toInt(&ok);
                if(!ok)
                {
                    hwError("FastFrame count query returned an invalid response."_L1);
                    hwDebug(u"%1: FastFrame count query returned an invalid response. Response = %2 (Hex: %3)"_s
                                .arg(d_key, QString(resp), QString(resp.toHex())));
                    return false;
                }
                if(n != numFrames)
                {
                    hwError(u"Requested number of FastFrames (%1) is different than actual number (%2)."_s
                                .arg(numFrames).arg(n));
                    return false;
                }
            }
            else
            {
                hwError("Gave an empty response to FastFrame count query."_L1);
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
                    hwError(u"Could not configure FastFrame summary frame to %1."_s.arg(sumfConfig));
                    hwDebug(u"%1: Could not configure FastFrame summary frame to %2. Response = %3 (Hex: %4)"_s
                                .arg(d_key, sumfConfig, QString(resp), QString(resp.toHex())));
                    return false;
                }
            }
            else
            {
                hwError("Gave an empty response to FastFrame summary frame query."_L1);
                return false;
            }

            if(config.d_blockAverage)
            {
                //this forces the scope to only return the final frame, which is the summary frame
                if(!p_comm->writeCmd(QString(":DATA:FRAMESTART 100000;FRAMESTOP 100000\n")))
                {
                    hwError("Could not configure summary frame."_L1);
                    return false;
                }
            }
            else
            {
                //this forces the scope to return all frames
                if(!p_comm->writeCmd(QString(":DATA:FRAMESTART 1;FRAMESTOP 100000\n")))
                {
                    hwError("Could not configure frames."_L1);
                    return false;
                }
            }
        }
        else
        {
            hwError("Gave an empty response to FastFrame max frames query."_L1);
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
            hwError("Could not verify trigger channel."_L1);
            hwDebug(u"%1: Could not verify trigger channel. Response = %2 (Hex: %3)"_s
                        .arg(d_key, QString(resp), QString(resp.toHex())));
            return false;
        }

        if(!QString(resp).contains(slope,Qt::CaseInsensitive))
        {
            hwError("Could not verify trigger slope."_L1);
            hwDebug(u"%1: Could not verify trigger slope. Response = %2 (Hex: %3)"_s
                        .arg(d_key, QString(resp), QString(resp.toHex())));
            return false;
        }
    }
    else
    {
        hwError("Gave an empty response to trigger query."_L1);
        return false;
    }

    //set waveform output settings
    if(config.d_blockAverage)
    {
        if(config.d_bytesPerPoint != 2)
            hwWarn("Settting bytes per point to 2 for averaging"_L1);
        config.d_bytesPerPoint = 2;
        resp = p_comm->queryCmd(QString(":HORIZONTAL:FASTFRAME:SIXTEENBIT ON;SIXTEENBIT?\n"));
        if(!resp.contains("1"))
        {
            hwError("Could not configure scope for 16-bit summary frame"_L1);
            return false;
        }
    }
    else
        p_comm->writeCmd(QString(":HORIZONTAL:FASTFRAME:SIXTEENBIT OFF\n"));

    if(!p_comm->writeCmd(QString(":WFMOUTPRE:ENCDG BIN;BN_FMT RI;BYT_OR LSB;BYT_NR %1\n").arg(config.d_bytesPerPoint)))
    {
        hwError("Could not send waveform output commands."_L1);
        return false;
    }

    //acquisition settings
    if(!p_comm->writeCmd(QString(":ACQUIRE:MODE SAMPLE;STOPAFTER RUNSTOP;STATE RUN\n")))
    {
        hwError("Could not send acquisition commands."_L1);
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
            hwError("Could not parse response to waveform output settings query."_L1);
            hwDebug(u"%1: Could not parse response to waveform output settings query. Response = %2 (Hex: %3)"_s
                        .arg(d_key, QString(resp), QString(resp.toHex())));
            return false;
        }

        //check encoding
        if(!l.at(0).contains(QString("BIN"),Qt::CaseInsensitive))
        {
            hwError("Waveform encoding could not be set to binary."_L1);
            hwDebug(u"%1: Waveform encoding could not be set to binary. Response = %2 (Hex: %3)"_s
                        .arg(d_key, l.at(0), QString(l.at(0).toLatin1().toHex())));
            return false;
        }
        //check binary format
        if(!l.at(1).contains(QString("RI"),Qt::CaseInsensitive))
        {
            hwError("Waveform format could not be set to signed integer."_L1);
            hwDebug(u"%1: Waveform format could not be set to signed integer. Response = %2 (Hex: %3)"_s
                        .arg(d_key, l.at(1), QString(l.at(1).toLatin1().toHex())));
            return false;
        }
        //check byte order
        if(!l.at(2).contains(QString("LSB"),Qt::CaseInsensitive))
        {
            hwError("Waveform format could not be set to least significant byte first."_L1);
            hwDebug(u"%1: Waveform format could not be set to least significant byte first. Response = %2 (Hex: %3)"_s
                        .arg(d_key, l.at(2), QString(l.at(2).toLatin1().toHex())));
            return false;
        }
        config.d_byteOrder = DigitizerConfig::LittleEndian;

        //verify number of frames
        if(d_multiRecord)
        {
            if(l.at(3).toInt() != config.d_numRecords)
            {
                hwError(u"Waveform contains the wrong number of frames. Target: %1, Actual: %2."_s
                            .arg(config.d_numRecords).arg(l.at(3).toInt()));
                hwDebug(u"%1: Waveform contains the wrong number of frames. Target: %2, Actual: %3. Response = %4 (Hex: %5)"_s
                            .arg(d_key).arg(config.d_numRecords).arg(l.at(3).toInt()).arg(l.at(3), QString(l.at(3).toLatin1().toHex())));
                return false;
            }
        }
        else if (l.at(3).toInt() != 1)
        {
            hwError(u"Waveform contains the wrong number of frames. Target: 1, Actual: %1."_s.arg(l.at(3).toInt()));
            hwDebug(u"%1: Waveform contains the wrong number of frames. Target: 1, Actual: %2. Response = %3 (Hex: %4)"_s
                        .arg(d_key).arg(l.at(3).toInt()).arg(l.at(3), QString(l.at(3).toLatin1().toHex())));
            return false;
        }
        //verify record length
        bool ok = false;
        int recLen = l.at(4).toInt(&ok);
        if(!ok)
        {
            hwError("Could not parse waveform record length response."_L1);
            hwDebug(u"%1: Could not parse waveform record length response. Response = %2 (Hex: %3)"_s
                        .arg(d_key, l.at(4), QString(l.at(4).toLatin1().toHex())));
            return false;
        }
        if(recLen != config.d_recordLength)
        {
            hwWarn(u"Record length is %1. Requested value was %2. Proceeding with %1 samples."_s
                       .arg(recLen).arg(config.d_recordLength));
            config.d_recordLength = recLen;
        }
        //verify byte number
        int bpp = l.at(5).mid(0,1).toInt(&ok);
        if(!ok || bpp != config.d_bytesPerPoint)
        {
            hwError("Invalid response to bytes per point query."_L1);
            hwDebug(u"%1: Invalid response to bytes per point query. Response = %2 (Hex: %3)"_s
                        .arg(d_key, l.at(8), QString(l.at(8).toLatin1().toHex())));
            return false;
        }
    }

    auto cfg = dynamic_cast<FtmwDigitizerConfig*>(this);
    if(cfg)
        *cfg = config;
    else
    {
        hwError("Could not record digitizer config settings."_L1);
        return false;
    }

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
//    hwDebug(u"%1: Bytes available: %2\t%3 ms"_s.arg(d_key).arg(ba).arg(QTime::currentTime().msec()));

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
//                hwDebug(u"%1: Found hdr: %2 ms"_s.arg(d_key).arg(QTime::currentTime().msec()));
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
//            hwDebug(u"%1: Wfm read complete: %2 ms"_s.arg(d_key).arg(QTime::currentTime().msec()));
            emitShot(wfm);
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
    hwWarn("Attempting to wake up scope."_L1);

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
    hwError(u"Socket error: %1 - %2"_s.arg((int)e).arg(p_socket->errorString()));
    emit hardwareFailure();
}

QByteArray MSO72004C::scopeQueryCmd(const QString &query)
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
