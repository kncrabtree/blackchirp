#include "m4i2220x8.h"

#include <math.h>


M4i2220x8::M4i2220x8(QObject *parent) : FtmwScope(parent), p_handle(nullptr)
{
    d_subKey = QString("m4i2220x8");
    d_prettyName = QString("Spectrum Instrumentation M4i.2220-x8 Digitizer");
    d_commType = CommunicationProtocol::Custom;
    d_threaded = true;



    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    s.setValue(QString("canBlockAverage"),true);
    s.setValue(QString("canFastFrame"),true);
    s.setValue(QString("canSummaryFrame"),false);
    s.setValue(QString("canBlockAndFastFrame"),false);

    double bandwidth = s.value(QString("bandwidth"),1250.0).toDouble();
    s.setValue(QString("bandwidth"),bandwidth);

    if(s.beginReadArray(QString("sampleRates")) < 1)
    {
        s.endArray();

        QList<QPair<QString,double>> sampleRates;
        for(int i=0; i<6; i++)
        {
            QString txt = QString("%1 MSa/S").arg(round(2.5e3/( 1 << i )),4);
            double val = 2.5e9/(static_cast<double>( 1 << i));
            sampleRates << qMakePair(txt,val);
        }

        s.beginWriteArray(QString("sampleRates"));
        for(int i=0; i<sampleRates.size(); i++)
        {
            s.setArrayIndex(i);
            s.setValue(QString("text"),sampleRates.at(i).first);
            s.setValue(QString("val"),sampleRates.at(i).second);
        }
        s.endArray();
    }

    s.beginWriteArray(QString("comm"));
    s.setArrayIndex(0);
    s.setValue(QString("name"),QString("Device Path"));
    s.setValue(QString("key"),QString("devPath"));
    s.setValue(QString("type"),QString("string"));
    s.endArray();

    s.endGroup();
    s.endGroup();

    p_timer = new QTimer(this);
}

M4i2220x8::~M4i2220x8()
{
    if(p_handle != nullptr)
    {
        spcm_vClose(p_handle);
        p_handle = nullptr;
    }
}


bool M4i2220x8::testConnection()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    QByteArray path = s.value(QString("devPath"),QString("/dev/spcm0")).toString().toLatin1();
    s.endGroup();
    s.endGroup();

    if(p_handle != nullptr)
    {
        spcm_vClose(p_handle);
        p_handle = nullptr;
    }

    p_handle = spcm_hOpen(path.data());


    if(p_handle == nullptr)
    {
        emit connected(false,QString("Could not connect to digitizer. Verify that %1 exists and is accessible.").arg(QString(path)));
        return false;
    }

    qint32 cType = 0;
    spcm_dwGetParam_i32(p_handle,SPC_PCITYP,&cType);

    qint32 serialNo = 0;
    spcm_dwGetParam_i32(p_handle,SPC_PCISERIALNO,&serialNo);

    qint32 driVer = 0;
    spcm_dwGetParam_i32(p_handle,SPC_GETDRVVERSION,&driVer);

    qint32 kerVer = 0;
    spcm_dwGetParam_i32(p_handle,SPC_GETKERNELVERSION,&kerVer);

    QByteArray errText(1000,'\0');
    if(spcm_dwGetErrorInfo_i32(p_handle,NULL,NULL,errText.data()) != ERR_OK)
    {
        emit connected(false, QString::fromLatin1(errText));
        return false;
    }

    emit logMessage(QString("Card type: %1, Serial Number %2. Library V %3.%4 build %5. Kernel V %6.%7 build %8")
                    .arg(cType).arg(serialNo).arg(driVer >> 24).arg((driVer >> 16) & 0xff).arg(driVer & 0xffff)
                    .arg(kerVer >> 24).arg((kerVer >> 16) & 0xff).arg(kerVer & 0xffff));

    emit connected();
    return true;
}

void M4i2220x8::initialize()
{
    testConnection();
}

Experiment M4i2220x8::prepareForExperiment(Experiment exp)
{
    if(!exp.ftmwConfig().isEnabled())
    {
        d_enabledForExperiment = false;
        return exp;
    }

    d_enabledForExperiment = true;

    //first, reset the card so all registers are in default states
    spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_RESET);

    auto sc = exp.ftmwConfig().scopeConfig();

    //this card only has 1 channel, so enable it and disable all others regardless of user's entry
    if(sc.fidChannel != 1)
        emit logMessage(QString("FID channel set to 1 (selected: %1) because the device only has a single channel.").arg(sc.fidChannel),BlackChirp::LogWarning);
    spcm_dwSetParam_i32(p_handle,SPC_CHENABLE,CHANNEL0);
    sc.fidChannel = 1;

    //4 possible input ranges: 200, 500, 1000, or 2500 mV.
    //If user has chosen something else, select the nearest range and issue a warning
    qint32 range = 0;
    qint32 r = static_cast<qint32>(round(sc.vScale*1000.0));

    if(r == 200)
    {
        sc.vScale = 0.2;
        range = 200;
    }
    else if(r < 350)
    {
        emit logMessage(QString("Input range set to 200 mV (selected: %1 mV). Valid values are 200, 500, 1000, and 2500 mV.").arg(r));
        sc.vScale = 0.2;
        range = 200;
    }
    else if(r == 500)
    {
        sc.vScale = 0.5;
        range = 500;
    }
    else if(r < 750)
    {
        emit logMessage(QString("Input range set to 500 mV (selected: %1 mV). Valid values are 200, 500, 1000, and 2500 mV.").arg(r));
        sc.vScale = 0.5;
        range = 500;
    }
    else if(r == 1000)
    {
        sc.vScale = 1.0;
        range = 1000;
    }
    else if(r < 1750)
    {
        emit logMessage(QString("Input range set to 1000 mV (selected: %1 mV). Valid values are 200, 500, 1000, and 2500 mV.").arg(r));
        sc.vScale = 1.0;
        range = 1000;
    }
    else if(r == 2500)
    {
        sc.vScale = 2.5;
        range = 2500;
    }
    else
    {
        emit logMessage(QString("Input range set to 2500 mV (selected: %1 mV). Valid values are 200, 500, 1000, and 2500 mV.").arg(r));
        sc.vScale = 2.5;
        range = 2500;
    }

    spcm_dwSetParam_i32(p_handle,SPC_AMP0,range);

    //set offset to 0
    spcm_dwSetParam_i32(p_handle,SPC_OFFS0,0);
    sc.vOffset = 0.0;

    //set to AC coupling
    spcm_dwSetParam_i32(p_handle,SPC_ACDC0,1);


    //Configure clock source
    spcm_dwSetParam_i32(p_handle,SPC_CLOCKMODE,SPC_CM_EXTREFCLOCK);
    spcm_dwSetParam_i32(p_handle,SPC_REFERENCECLOCK,1250000000);

    // Configure sample rate
    spcm_dwSetParam_i64(p_handle,SPC_SAMPLERATE,static_cast<qint64>(sc.sampleRate));

    //configure trigger
    if(sc.trigChannel != 0)
        emit logMessage(QString("Trigger channel set to External Input (selected: %1). Must trigger on Ext In."),BlackChirp::LogWarning);
    sc.trigChannel = 0;



    if(sc.slope == BlackChirp::RisingEdge)
        spcm_dwSetParam_i32(p_handle,SPC_TRIG_EXT0_MODE,SPC_TM_POS);
    else
        spcm_dwSetParam_i32(p_handle,SPC_TRIG_EXT0_MODE,SPC_TM_NEG);

    spcm_dwSetParam_i32(p_handle,SPC_TRIG_ORMASK,SPC_TMASK_EXT0);

//    spcm_dwSetParam_i32(p_handle,SPC_TRIG_TERM,1);
    spcm_dwSetParam_i32(p_handle,SPC_TRIG_EXT0_LEVEL0,static_cast<qint32>(round(sc.trigLevel*1000.0)));



    //configure acquisition for multi FIFO mode
    int dataWidth = 1;
    if(sc.blockAverageEnabled)
    {
        if(sc.numAverages <= 256)
        {
            dataWidth = 2;
            spcm_dwSetParam_i32(p_handle,SPC_CARDMODE,SPC_REC_FIFO_AVERAGE_16BIT);
        }
        else
        {
            dataWidth = 4;
            spcm_dwSetParam_i32(p_handle,SPC_CARDMODE,SPC_REC_FIFO_AVERAGE);
        }
        spcm_dwSetParam_i32(p_handle,SPC_AVERAGES,static_cast<qint32>(sc.numAverages));
        sc.fastFrameEnabled = false;
        sc.numFrames = 1;
    }
    else
    {
        spcm_dwSetParam_i32(p_handle,SPC_CARDMODE,SPC_REC_FIFO_MULTI);
        sc.numAverages = 1;
        if(!sc.fastFrameEnabled)
            sc.numFrames = 1;
    }

    //record length must be a multiple of 32
    if(sc.recordLength % 32)
    {
        sc.recordLength += (32 - (sc.recordLength % 32));
        emit logMessage(QString("Setting record length to %1 because it must be a multiple of 32.").arg(sc.recordLength),BlackChirp::LogWarning);
    }

    //configure record length
    spcm_dwSetParam_i64(p_handle,SPC_MEMSIZE,Q_UINT64_C(2147483648));
    spcm_dwSetParam_i64(p_handle,SPC_SEGMENTSIZE,static_cast<qint64>(sc.recordLength));
    spcm_dwSetParam_i64(p_handle,SPC_POSTTRIGGER,static_cast<qint64>(sc.recordLength-32));
    spcm_dwSetParam_i64(p_handle,SPC_LOOPS,static_cast<qint64>(16000000));

    d_bufferSize = sc.recordLength*dataWidth*sc.numFrames*10;

    sc.bytesPerPoint = dataWidth;

    d_waveformBytes = sc.recordLength*dataWidth*sc.numFrames;
    p_m4iBuffer = new char[d_bufferSize];

    spcm_dwDefTransfer_i64(p_handle,SPCM_BUF_DATA,SPCM_DIR_CARDTOPC,4096*4,static_cast<void*>(p_m4iBuffer),0,d_bufferSize);

    sc.yMult = sc.vScale/128.0;
    sc.byteOrder = QDataStream::LittleEndian;
    sc.vOffset = 0.0;
    sc.yOff = 0;
    sc.xIncr = 1.0/sc.sampleRate;


    QByteArray errText(1000,'\0');
    if(spcm_dwGetErrorInfo_i32(p_handle,NULL,NULL,errText.data()) != ERR_OK)
    {
        exp.setHardwareFailed();
        exp.setErrorString(QString("Could not initialize %1. Error message: %2").arg(d_prettyName).arg(QString::fromLatin1(errText)));
        spcm_dwInvalidateBuf(p_handle,SPCM_BUF_DATA);
        delete[] p_m4iBuffer;
    }

    exp.setScopeConfig(sc);
    return exp;
}

void M4i2220x8::beginAcquisition()
{
    if(d_enabledForExperiment)
    {
        spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER);
        spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_DATA_STARTDMA);

        QByteArray errText(1000,'\0');
        if(spcm_dwGetErrorInfo_i32(p_handle,NULL,NULL,errText.data()) != ERR_OK)
        {
            emit logMessage(QString::fromLatin1(errText),BlackChirp::LogError);
            emit hardwareFailure();
            return;
        }

        connect(p_timer,&QTimer::timeout,this,&M4i2220x8::readWaveform,Qt::UniqueConnection);
        p_timer->start(100);
    }
}

void M4i2220x8::endAcquisition()
{
    if(d_enabledForExperiment)
    {
        p_timer->stop();
        disconnect(p_timer,&QTimer::timeout,this,&M4i2220x8::readWaveform);

        spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_STOP);
        spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_DATA_STOPDMA);
        spcm_dwInvalidateBuf(p_handle,SPCM_BUF_DATA);
        delete[] p_m4iBuffer;

        d_waveformBytes = 0;
    }

}

void M4i2220x8::readTimeData()
{
}

void M4i2220x8::readWaveform()
{
    //check to see if a data block is ready
    qint32 stat = 0;
    spcm_dwGetParam_i32(p_handle,SPC_M2STATUS,&stat);

    if(stat & M2STAT_DATA_ERROR) //internal error
    {
        QByteArray errText(1000,'\0');
        if(spcm_dwGetErrorInfo_i32(p_handle,NULL,NULL,errText.data()) != ERR_OK)
            emit logMessage(QString::fromLatin1(errText),BlackChirp::LogError);

        emit hardwareFailure();
        p_timer->stop();
        disconnect(p_timer,&QTimer::timeout,this,&M4i2220x8::readWaveform);
        return;
    }

    if(stat & M2STAT_DATA_BLOCKREADY) //block is ready
    {
        //read how many bytes are available
        qint64 ba = 0;
        spcm_dwGetParam_i64(p_handle,SPC_DATA_AVAIL_USER_LEN,&ba);

        //are enough bytes available to make a waveform?
        if(ba >= d_waveformBytes || stat & M2STAT_DATA_END)
        {
            //transfer data into a QByteArray
            QByteArray out;
            out.resize(d_waveformBytes);

            //get current position
            qint64 pos = 0;
            spcm_dwGetParam_i64(p_handle,SPC_DATA_AVAIL_USER_POS,&pos);

            //copy data; rolling over at end of buffer if necessary
            for(int i=0; i<d_waveformBytes; i++)
            {
                int thisPos = i + pos;
                if(thisPos >= d_bufferSize)
                    thisPos -= d_bufferSize;

                out[i] = p_m4iBuffer[thisPos];
            }

            //tell m4i that it can use this memory again
            spcm_dwSetParam_i64(p_handle,SPC_DATA_AVAIL_CARD_LEN,d_waveformBytes);

            emit shotAcquired(out);
        }
    }

    //workaround for SPC_LOOPS issue
    if(stat & M2STAT_DATA_END)
    {
        spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_STOP);
        spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_DATA_STOPDMA);

        spcm_dwInvalidateBuf(p_handle,SPCM_BUF_DATA);

        spcm_dwDefTransfer_i64(p_handle,SPCM_BUF_DATA,SPCM_DIR_CARDTOPC,4096*4,static_cast<void*>(p_m4iBuffer),0,d_bufferSize);



        spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER);
        spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_DATA_STARTDMA);

        QByteArray errText(1000,'\0');
        if(spcm_dwGetErrorInfo_i32(p_handle,NULL,NULL,errText.data()) != ERR_OK)
        {
            emit logMessage(QString::fromLatin1(errText),BlackChirp::LogError);
            emit hardwareFailure();
            return;
        }
    }
}
