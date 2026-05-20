#include "m8190.h"
#include <hardware/core/hardwareregistration.h>

#include <math.h>

// Register hardware implementation
REGISTER_HARDWARE_META(M8190, "Keysight M8190 AWG")
REGISTER_HARDWARE_PROTOCOLS(M8190, CommunicationProtocol::Tcp)
REGISTER_HARDWARE_SETTINGS(M8190,
    {BC::Key::AWG::rate, "Sample Rate (Hz)", "DAC output sample rate",
     9.375e9, 1e6, 1000e9, HwSettingPriority::Important},
    {BC::Key::AWG::min, "Min Freq (MHz)", "Minimum chirp frequency in MHz",
     0.0, 0.0, QVariant{}, HwSettingPriority::Important},
    {BC::Key::AWG::max, "Max Freq (MHz)", "Maximum chirp frequency in MHz",
     5000.0, 0.0, QVariant{}, HwSettingPriority::Important},
    {BC::Key::AWG::markerCount, "Marker Count", "Number of physical marker output channels",
     2, 0, QVariant{}, HwSettingPriority::Required}
)

M8190::M8190(const QString& label, QObject *parent) : AWG(QString(M8190::staticMetaObject.className()), label, parent)
{
    save();
}


bool M8190::testConnection()
{
    QByteArray resp = p_comm->queryCmd(QString("*IDN?\n"));

    if(resp.isEmpty())
    {
        d_errorString = QString("Did not respond to ID query.");
        return false;
    }

    if(!resp.startsWith(QByteArray("Agilent Technologies,M8190")))
    {
        d_errorString = QString("ID response invalid. Response: %1 (Hex: %2)").arg(QString(resp.trimmed())).arg(QString(resp.toHex()));
        return false;
    }

    hwDebug(u"ID response: %1"_s.arg(QString(resp.trimmed())));

    p_comm->writeCmd(QString("*CLS\n"));

    return true;
}

void M8190::initialize()
{
}

bool M8190::prepareForExperiment(Experiment &exp)
{
    bool triggered = get<bool>(BC::Key::AWG::triggered);
    double samplerate = get<double>(BC::Key::AWG::rate);
    Q_UNUSED(samplerate)

    d_enabledForExperiment = exp.ftmwEnabled();
    if(!d_enabledForExperiment)
        return true;

    p_comm->writeCmd(QString("*CLS;*RST\n"));

    if(!m8190Write(QString(":INST:COUP:STAT ON\n"))) // channel coupling
    {
        exp.d_errorString = QString("Could not set channel coupling on.");
        return false;
    }

    //external reference (TODO: interface with more general clock system?)
        if(!m8190Write(QString(":ROSC:SOUR EXT;:ROSC:FREQ 10000000;:FREQ:RAST 9375000000 \n"))) // for 10 MHz ref clock in
    {
        exp.d_errorString = QString("Could not set to external reference clock.");
        return false;
    }

    if(!m8190Write(QString(":FREQ:RAST:EXT 9375000000;:FREQ:RAST:SOURCE EXT\n"))) // for sample clock in
    {
        exp.d_errorString = QString("Could not set to external sample clock.");
        return false;
    }

    //external triggering
    if(triggered)
    {
        QString trig("ASYN");
//        if(asyn)
//            trig.prepend(QString("A"));

        if(!m8190Write(QString(":INIT:CONT:STAT 0;:INIT:GATE 0;:ARM:TRIG:SOUR EXT;:ARM:TRIG:LEV 0.5;:ARM:TRIG:SLOP POS\n")))
        {
            exp.d_errorString = QString("Could not initialize trigger settings.");
            return false;
        }
    }
    else
    {
        if(!m8190Write(QString(":INIT:CONT 1;:INIT:GATE 0\n")))
        {
            exp.d_errorString = QString("Could not initialize continuous signal generation.");
            return false;
        }
    }

//    if(!m8190Write(QString(":VOLTAGE 0.675\n")))
    if(!m8190Write(QString(":DAC:VOLT:AMPL 0.675\n")))
    {
        exp.d_errorString = QString("Could not set output voltage.");
        return false;
    }

    if(!m8190Write(QString(":TRAC:DEL:ALL\n")))
    {
        exp.d_errorString = QString("Could not delete old traces.");
        return false;
    }

    auto data = exp.ftmwConfig()->d_rfConfig.d_chirpConfig.getChirpMicroseconds();
    auto packedMarkers = exp.ftmwConfig()->d_rfConfig.d_chirpConfig.getPackedMarkerData();

    //use speed mode, 12-bit
    if(!m8190Write(QString(":TRAC1:DWIDth WSPeed;:TRAC2:DWIDth WSPeed\n")))
    {
        exp.d_errorString = QString("Could not initialize 12-bit DAC resolution.");
        return false;
    }

    int pad = (256 - (data.size()%256))%256;
    int len = data.size() + pad;

    if(len<320)
    {
        exp.d_errorString = QString("Not enough samples!");
        return false;
    }

    QByteArray id = p_comm->queryCmd(QString(":TRAC1:DEF:NEW? %1\n").arg(len)).trimmed();
    if(id.isEmpty())
    {
        exp.d_errorString = QString("Could not create new AWG trace.");
        return false;
    }

    //each transfer must align with 256-sample memory vectors
    int chunkSize = 1 << 12;
    int chunks = static_cast<int>(ceil(static_cast<double>(data.size())/static_cast<double>(chunkSize)));
    int currentChunk = 0;

    QByteArray chunkData;
    chunkData.reserve(chunkSize*2);
    bool success = true;

    while(currentChunk < chunks)
    {
        chunkData.clear();
        int startIndex = currentChunk*chunkSize;
        //if this chunk runs past the data size, pad with zeros until we reach nearest
        //multiple of 256
        int endIndex = qMin((currentChunk+1)*chunkSize,data.size()+pad);
        int numPnts = endIndex - startIndex;

        //AWG has analog and marker values interleaved
        for(int i=0; i < numPnts; i++)
        {
            qint16 low = -2047;
            qint16 high = 2047;
            //convert doubles to qint16: -1.0 --> -2047, +1.0 --> 2047
            qint16 chirpVal = 0;
            if(startIndex + i < data.size())
                chirpVal = qBound(low,static_cast<qint16>(round(data.at(startIndex+i).y()*2047.0)),high);

            chirpVal = (chirpVal<<4);

//            markers are 2 last bits: sync marker is second to last and sample marker is very last bit

            if(chirpVal!=0)
            {
                chirpVal += 1; // will fill last bit with 1
            }

//            qint16 markerVal = 0;
//            if(startIndex + i < data.size())
//            {
//                if(markerData.at(startIndex+i).first) //marker 1 (protection)
//                    markerVal+=1;
//            }

//            chirpVal = chirpVal + markerVal;
            chunkData.append(chirpVal);
        }

        //create data header

        QString header = QString(":TRAC1:DATA %1,%2,").arg(QString(id)).arg(startIndex);

        QString binSize = QString::number(numPnts*2);
        QString binHeader = QString("#%1%2").arg(binSize.size()).arg(binSize);
        header.append(binHeader);

        if(!p_comm->writeCmd(header))
        {
            success = false;
            exp.d_errorString = QString("Could not write header data to AWG. Header: %1").arg(header);
            break;
        }

        if(!p_comm->writeBinary(chunkData))
        {
            success = false;
            exp.d_errorString = QString("Could not write waveform data to AWG. Header was: %1").arg(header);
            break;
        }

        p_comm->writeCmd(QString("\n"));

        if(!m8190Write(QString(":TRAC1:SEL 1\n")))
        {
            exp.d_errorString = QString("Could not select segment.");
            return false;
        }

        QByteArray resp = p_comm->queryCmd(QString(":SYST:ERR?\n"));
        if(!resp.startsWith('0'))
        {
            exp.d_errorString = QString("Could not write waveform data to AWG. Error %1. Header was: %2").arg(QString(resp)).arg(header);
            success = false;
            break;
        }

        currentChunk++;
    }

    return success;

}

void M8190::beginAcquisition()
{
    if(d_enabledForExperiment)
    {
        p_comm->writeCmd(QString(":OUTP1 ON;:TRAC1:MARK OFF;:OUTP2 OFF;:TRAC2:MARK OFF\n")); //;:TRAC2:MARK ON\n"));
        p_comm->writeCmd(QString(":INIT:IMM\n"));
    }
}

void M8190::endAcquisition()
{
    if(d_enabledForExperiment)
    {
        p_comm->writeCmd(QString(":OUTP1 OFF;:TRAC1:MARK OFF;:OUTP2 OFF;:TRAC2:MARK OFF\n"));
        p_comm->writeCmd(QString(":FREQ:RAST 9375000000;:FREQ:RAST:SOUR INT\n")); // for 10 MHz ref clock in
        p_comm->writeCmd(QString(":ABOR\n"));
    }
}

bool M8190::m8190Write(const QString &cmd)
{
    if(!p_comm->writeCmd(cmd))
        return false;

    QByteArray resp = p_comm->queryCmd(QString(":SYST:ERR?\n"));
    if(!resp.startsWith('0'))
    {
       hwError(u"Could not write waveform data to AWG. Error: %1. Command was: %2"_s.arg(QString(resp), cmd));
        return false;
    }

    return true;
}
