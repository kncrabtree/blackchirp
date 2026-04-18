#include "dsox92004a.h"

#include <QTcpSocket>
#include <QTimer>
#include <hardware/core/hardwareregistration.h>

using namespace BC::Key::FtmwScope;
using namespace BC::Key::Digi;

// Register this hardware implementation
REGISTER_HARDWARE_META(DSOx92004A, "Keysight DSOx92004A FTMW Digitizer (20 GHz, 80 GS/s)")
REGISTER_HARDWARE_PROTOCOLS(DSOx92004A, CommunicationProtocol::Tcp)
REGISTER_HARDWARE_SETTINGS(DSOx92004A,
    {numAnalogChannels,  "Analog Channels",  "Number of analog inputs",
     4, 1, 32, HwSettingPriority::Required},
    {numDigitalChannels, "Digital Channels",  "Number of digital inputs",
     0, 0, 32, HwSettingPriority::Required},
    {hasAuxTriggerChannel, "Aux Trigger Channel", "Has auxiliary trigger input",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {minFullScale,       "Min Full Scale (V)", "Minimum full scale voltage",
     5e-2, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxFullScale,       "Max Full Scale (V)", "Maximum full scale voltage",
     2.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {minVOffset,         "Min V Offset (V)",   "Minimum voltage offset",
     -2.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxVOffset,         "Max V Offset (V)",   "Maximum voltage offset",
     2.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {isTriggered,        "Triggered",          "Digitizer uses external trigger",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {minTrigDelay,       "Min Trig Delay (us)", "Minimum trigger delay",
     -10.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxTrigDelay,       "Max Trig Delay (us)", "Maximum trigger delay",
     10.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {minTrigLevel,       "Min Trig Level (V)",  "Minimum trigger level",
     -5.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxTrigLevel,       "Max Trig Level (V)",  "Maximum trigger level",
     5.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxRecordLength,    "Max Record Length",   "Maximum record length in samples",
     100000000, 0, QVariant{}, HwSettingPriority::Optional},
    {canBlockAverage,    "Block Average",       "Supports block averaging",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxAverages,        "Max Averages",        "Maximum number of averages",
     100, 1, QVariant{}, HwSettingPriority::Optional},
    {canMultiRecord,     "Multi Record",        "Supports multi-record acquisition",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxRecords,         "Max Records",         "Maximum number of records",
     100, 1, QVariant{}, HwSettingPriority::Optional},
    {multiBlock,         "Multi Block",         "Can block average and multi-record simultaneously",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {maxBytes,           "Max Bytes/Point",     "Maximum bytes per data point",
     2, 1, 8, HwSettingPriority::Optional},
    {bandwidth,          "Bandwidth (MHz)",     "Analog bandwidth",
     20000.0, QVariant{}, QVariant{}, HwSettingPriority::Important}
)
REGISTER_HARDWARE_ARRAY(DSOx92004A, sampleRates,
    "Sample Rates", "Available digitizer sample rates",
    HwSettingPriority::Important)
REGISTER_HARDWARE_ARRAY_ENTRY(DSOx92004A, sampleRates,
    {{srText, "1 GSa/s"}, {srValue, 1e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(DSOx92004A, sampleRates,
    {{srText, "1.25 GSa/s"}, {srValue, 1.25e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(DSOx92004A, sampleRates,
    {{srText, "2 GSa/s"}, {srValue, 2e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(DSOx92004A, sampleRates,
    {{srText, "2.5 GSa/s"}, {srValue, 2.5e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(DSOx92004A, sampleRates,
    {{srText, "4 GSa/s"}, {srValue, 4e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(DSOx92004A, sampleRates,
    {{srText, "10 GSa/s"}, {srValue, 10e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(DSOx92004A, sampleRates,
    {{srText, "20 GSa/s"}, {srValue, 20e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(DSOx92004A, sampleRates,
    {{srText, "40 GSa/s"}, {srValue, 40e9}})
REGISTER_HARDWARE_ARRAY_ENTRY(DSOx92004A, sampleRates,
    {{srText, "80 GSa/s"}, {srValue, 80e9}})

DSOx92004A::DSOx92004A(const QString& label, QObject *parent) :
    FtmwScope(QString(DSOx92004A::staticMetaObject.className()), label, parent)
{

    // Communication defaults
    setDefault(BC::Key::Comm::timeout, 1000);
    setDefault(BC::Key::Comm::termChar, QString("\n"));

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

    hwDebug(u"%1: ID response: %2"_s.arg(d_key, QString(resp)));
    return true;

}

void DSOx92004A::initialize()
{
    p_socket = p_comm->device<QTcpSocket>();
    p_socket->setSocketOption(QAbstractSocket::LowDelayOption,1);

}

bool DSOx92004A::prepareForExperiment(Experiment &exp)
{
    d_enabledForExperiment = exp.ftmwEnabled();
    if(!d_enabledForExperiment)
        return true;

    auto &config = exp.ftmwConfig()->scopeConfig();

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

    p_comm->_device()->readAll();

    bool done = false;
    while(!done)
    {
        QByteArray resp = p_comm->queryCmd(QString(":SYSTEM:ERROR? STRING\n"));
        if(resp.startsWith('0') || resp.isEmpty())
            break;

        hwLog(resp);

    }

    //verify that FID channel was set correctly
    QByteArray resp = p_comm->queryCmd(QString(":WAVEFORM:SOURCE?\n"));
    if(resp.isEmpty() || !resp.contains(QString("CHAN%1").arg(config.d_fidChannel).toLatin1()))
    {
        hwError("Failed to set FID channel."_L1);
        hwDebug(u"%1: Failed to set FID channel. Response = %2 (Hex: %3)"_s
                    .arg(d_key, QString(resp), QString(resp.toHex())));
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
            hwError("Could not parse offset response."_L1);
            hwDebug(u"%1: Could not parse offset response. Response = %2 (Hex: %3)"_s
                        .arg(d_key, QString(resp), QString(resp.toHex())));
            return false;
        }
        config.d_analogChannels[d_fidChannel].offset = offset;
    }
    else
    {
        hwError("Gave an empty response to offset query."_L1);
        return false;
    }
    resp = p_comm->queryCmd(QString(":CHAN%1:SCALE?\n").arg(config.d_fidChannel));
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
        if(!(fabs(config.d_analogChannels[d_fidChannel].fullScale-scale*5.0) < 0.01))
            hwWarn(u"Vertical scale is different than specified. Target: %1 V, Scope setting: %2 V"_s
                       .arg(QString::number(config.d_analogChannels[d_fidChannel].fullScale,'f',3))
                       .arg(QString::number(scale*5.0,'f',3)));
        config.d_analogChannels[d_fidChannel].fullScale = scale*5.0;
    }
    else
    {
        hwError("Gave an empty response to scale query."_L1);
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
            hwError("Sample rate query returned an invalid response."_L1);
            hwDebug(u"%1: Sample rate query returned an invalid response. Response = %2 (Hex: %3)"_s
                        .arg(d_key, QString(resp), QString(resp.toHex())));
            return false;
        }
        if(!(fabs(sRate - config.d_sampleRate)<1e6))
        {
            hwError(u"Could not set sample rate successfully. Target: %1 GS/s, Scope setting: %2 GS/s"_s
                        .arg(QString::number(config.d_sampleRate/1e9,'f',3))
                        .arg(QString::number(sRate/1e9,'f',3)));
            return false;
        }
        config.d_sampleRate = sRate;
    }
    else
    {
        hwError("Gave an empty response to sample rate query."_L1);
        return false;
    }

    resp = p_comm->queryCmd(QString(":ACQUIRE:POINTS?\n"));
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
            hwWarn(u"Record length limited by scope memory. Length will be different than requested. Target: %1, Scope setting: %2"_s
                       .arg(config.d_recordLength).arg(recLength));
        }
        config.d_recordLength = recLength;
    }
    else
    {
        hwError("Gave an empty response to record length query."_L1);
        return false;
    }

    resp = p_comm->queryCmd(QString(":TRIGGER:EDGE:SOURCE?\n"));
    if(resp.isEmpty() || !QString(resp).contains(trigCh),Qt::CaseInsensitive)
    {
        hwError("Could not verify trigger channel."_L1);
        hwDebug(u"%1: Could not verify trigger channel. Response = %2 (Hex: %3)"_s
                    .arg(d_key, QString(resp), QString(resp.toHex())));
        return false;
    }


    resp = p_comm->queryCmd(QString(":TRIGGER:EDGE:SLOPE?\n"));
    if(resp.isEmpty() || !QString(resp).contains(slope))
    {
        hwError("Could not verify trigger slope."_L1);
        hwDebug(u"%1: Could not verify trigger slope. Response = %2 (Hex: %3)"_s
                    .arg(d_key, QString(resp), QString(resp.toHex())));
        return false;
    }

    auto cfg = dynamic_cast<FtmwDigitizerConfig*>(this);
    if(cfg)
        *cfg = config;
    else
    {
        hwError("Could not record digitizer config settings."_L1);
        return false;
    }

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
    emitShot(out);

    p_socket->readAll();

    connect(p_socket,&QTcpSocket::readyRead,this,&DSOx92004A::readWaveform);
//    p_comm->writeCmd(QString(":DIGITIZE;*OPC?\n"));
    p_comm->writeCmd(QString("*OPC?\n"));
//    p_queryTimer->start(100);

}

bool DSOx92004A::scopeCommand(const QString &cmd)
{
    QString orig = cmd;
    QString modified = cmd;
    if(modified.endsWith(QString("\n")))
        modified.chop(1);

    modified.append(QString(";:SYSTEM:ERROR?\n"));
    QByteArray resp = p_comm->queryCmd(modified,true);
    if(resp.isEmpty())
    {
        hwError(u"Timed out on query %1"_s.arg(orig));
        return false;
    }

    int val = resp.trimmed().toInt();
    if(val != 0)
    {
        hwError(u"Received error %1 on query %2"_s.arg(val).arg(orig));
        return false;
    }
    return true;
}
