#include "m4i2220x8.h"

#include <math.h>


M4i2220x8::M4i2220x8(QObject *parent) :
    FtmwScope(BC::Key::FtmwScope::m4i2220x8,BC::Key::FtmwScope::m4i2220x8Name,CommunicationProtocol::Custom,parent), p_handle(nullptr)
{
    setDefault(BC::Key::FtmwScope::blockAverage,true);
    setDefault(BC::Key::FtmwScope::multiRecord,true);
    setDefault(BC::Key::FtmwScope::summaryRecord,false);
    setDefault(BC::Key::FtmwScope::multiBlock,false);
    setDefault(BC::Key::FtmwScope::bandwidth,1250.0);

    if(!containsArray(BC::Key::FtmwScope::sampleRates))
        setArray(BC::Key::FtmwScope::sampleRates,{
                     {{BC::Key::FtmwScope::srText,"78.125 MSa/s"},{BC::Key::FtmwScope::srValue,2.5e9/32}},
                     {{BC::Key::FtmwScope::srText,"156.25 MSa/s"},{BC::Key::FtmwScope::srValue,2.5e9/16}},
                     {{BC::Key::FtmwScope::srText,"312.5 MSa/s"},{BC::Key::FtmwScope::srValue,2.5e9/8}},
                     {{BC::Key::FtmwScope::srText,"625 MSa/s"},{BC::Key::FtmwScope::srValue,2.5e9/4}},
                     {{BC::Key::FtmwScope::srText,"1250 GSa/s"},{BC::Key::FtmwScope::srValue,2.5e9/2}},
                     {{BC::Key::FtmwScope::srText,"2500 MSa/s"},{BC::Key::FtmwScope::srValue,2.5e9}}
                 });

    if(!containsArray(BC::Key::Custom::comm))
        setArray(BC::Key::Custom::comm, {
                    {{BC::Key::Custom::key,"devPath"},
                     {BC::Key::Custom::type,BC::Key::Custom::stringKey},
                     {BC::Key::Custom::label,"Device Path"}}
                 });
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
    auto path = getArrayValue(BC::Key::Custom::comm,0,"devPath",QString("/dev/spcm0"));

    if(p_handle != nullptr)
    {
        spcm_vClose(p_handle);
        p_handle = nullptr;
    }

    p_handle = spcm_hOpen(path.toLatin1().constData());


    if(p_handle == nullptr)
    {
        d_errorString = QString("Could not connect to digitizer. Verify that %1 exists and is accessible.").arg(QString(path));
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
        d_errorString =  QString::fromLatin1(errText);
        return false;
    }

    emit logMessage(QString("Card type: %1, Serial Number %2. Library V %3.%4 build %5. Kernel V %6.%7 build %8")
                    .arg(cType).arg(serialNo).arg(driVer >> 24).arg((driVer >> 16) & 0xff).arg(driVer & 0xffff)
                    .arg(kerVer >> 24).arg((kerVer >> 16) & 0xff).arg(kerVer & 0xffff));

    return true;
}

void M4i2220x8::initialize()
{
    p_timer = new QTimer(this);
}

bool M4i2220x8::prepareForExperiment(Experiment &exp)
{
    if(!exp.ftmwEnabled())
    {
        d_enabledForExperiment = false;
        return true;
    }

    d_enabledForExperiment = true;

    //first, reset the card so all registers are in default states
    spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_RESET);

    static_cast<FtmwDigitizerConfig>(*this) = exp.ftmwConfig()->d_scopeConfig;

    //this card only has 1 channel, so enable it and disable all others regardless of user's entry
    if(sc.fidChannel != 1)
        emit logMessage(QString("FID channel set to 1 (selected: %1) because the device only has a single channel.").arg(sc.fidChannel),LogHandler::Warning);
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
    auto clocks = exp.ftmwConfig()->d_rfConfig.getClocks();
    if(clocks.contains(RfConfig::DigRef) && !clocks.value(RfConfig::DigRef).hwKey.isEmpty())
    {
        spcm_dwSetParam_i32(p_handle,SPC_CLOCKMODE,SPC_CM_EXTREFCLOCK);
        spcm_dwSetParam_i32(p_handle,SPC_REFERENCECLOCK,qRound(clocks.value(BlackChirp::DigRef).desiredFreqMHz*1e6));
    }
    else
        spcm_dwSetParam_i32(p_handle,SPC_CLOCKMODE,SPC_CM_INTPLL);

    // Configure sample rate
    spcm_dwSetParam_i64(p_handle,SPC_SAMPLERATE,static_cast<qint64>(sc.sampleRate));

    //configure trigger
    if(sc.trigChannel != 0)
        emit logMessage(QString("Trigger channel set to External Input (selected: %1). Must trigger on Ext In."),LogHandler::Warning);
    sc.trigChannel = 0;




    spcm_dwSetParam_i32(p_handle,SPC_TRIG_ORMASK,SPC_TMASK_NONE);
    spcm_dwSetParam_i32(p_handle,SPC_TRIG_ORMASK,SPC_TMASK_EXT0);

    if(sc.slope == BlackChirp::RisingEdge)
        spcm_dwSetParam_i32(p_handle,SPC_TRIG_EXT0_MODE,SPC_TM_POS);
    else
        spcm_dwSetParam_i32(p_handle,SPC_TRIG_EXT0_MODE,SPC_TM_NEG);

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
        emit logMessage(QString("Setting record length to %1 because it must be a multiple of 32.").arg(sc.recordLength),LogHandler::Warning);
    }

    //configure record length
    spcm_dwSetParam_i64(p_handle,SPC_MEMSIZE,Q_UINT64_C(2147483648));
    spcm_dwSetParam_i64(p_handle,SPC_SEGMENTSIZE,static_cast<qint64>(sc.recordLength));
    spcm_dwSetParam_i64(p_handle,SPC_POSTTRIGGER,static_cast<qint64>(sc.recordLength-6400));
    spcm_dwSetParam_i64(p_handle,SPC_LOOPS,static_cast<qint64>(16000000));

    d_bufferSize = sc.recordLength*dataWidth*sc.numFrames*10;

    sc.bytesPerPoint = dataWidth;

    d_waveformBytes = sc.recordLength*dataWidth*sc.numFrames;
    p_m4iBuffer = new char[d_bufferSize];

    spcm_dwDefTransfer_i64(p_handle,SPCM_BUF_DATA,SPCM_DIR_CARDTOPC,4096*4,static_cast<void*>(p_m4iBuffer),0,d_bufferSize);

    sc.yMult = sc.vScale/128.0;
    sc.byteOrder = DigitizerConfig::LittleEndian;
    sc.vOffset = 0.0;
    sc.yOff = 0;
    sc.xIncr = 1.0/sc.sampleRate;


    QByteArray errText(1000,'\0');
    if(spcm_dwGetErrorInfo_i32(p_handle,NULL,NULL,errText.data()) != ERR_OK)
    {
        exp.setErrorString(QString("Could not initialize %1. Error message: %2").arg(d_name).arg(QString::fromLatin1(errText)));
        spcm_dwInvalidateBuf(p_handle,SPCM_BUF_DATA);
        delete[] p_m4iBuffer;
        return false;
    }

    exp.setScopeConfig(sc);
    return true;
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
            emit logMessage(QString::fromLatin1(errText),LogHandler::Error);
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

void M4i2220x8::readWaveform()
{
    //check to see if a data block is ready
    qint32 stat = 0;
    spcm_dwGetParam_i32(p_handle,SPC_M2STATUS,&stat);

    if(stat & M2STAT_DATA_ERROR) //internal error
    {
        QByteArray errText(1000,'\0');
        if(spcm_dwGetErrorInfo_i32(p_handle,NULL,NULL,errText.data()) != ERR_OK)
            emit logMessage(QString::fromLatin1(errText),LogHandler::Error);

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
            emit logMessage(QString::fromLatin1(errText),LogHandler::Error);
            emit hardwareFailure();
            return;
        }
    }
}
