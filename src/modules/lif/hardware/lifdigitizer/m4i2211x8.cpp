#include "m4i2211x8.h"

#include <QTimer>

M4i2211x8::M4i2211x8(QObject *parent) :
    LifScope (BC::Key::LifDigi::m4i2211x8,BC::Key::LifDigi::m4i2211x8Name,CommunicationProtocol::Custom,parent),
    p_handle(nullptr)
{
    using namespace BC::Key::Digi;
    setDefault(numAnalogChannels,2);
    setDefault(numDigitalChannels,0);
    setDefault(hasAuxTriggerChannel,true);
    setDefault(minFullScale,5e-2);
    setDefault(maxFullScale,2.5);
    setDefault(minVOffset,-2.0);
    setDefault(maxVOffset,2.0);
    setDefault(isTriggered,true);
    setDefault(minTrigDelay,-10.0);
    setDefault(maxTrigDelay,10.0);
    setDefault(minTrigLevel,-5.0);
    setDefault(maxTrigLevel,5.0);
    setDefault(canBlockAverage,false);
    setDefault(canMultiRecord,false);
    setDefault(multiBlock,false);
    setDefault(maxBytes,1);

    if(!containsArray(sampleRates))
        setArray(sampleRates,{
                     {{srText,"78.125 MSa/s"},{srValue,2.5e9/32}},
                     {{srText,"156.25 MSa/s"},{srValue,2.5e9/16}},
                     {{srText,"312.5 MSa/s"},{srValue,2.5e9/8}},
                     {{srText,"625 MSa/s"},{srValue,2.5e9/4}},
                     {{srText,"1250 GSa/s"},{srValue,2.5e9/2}},
                 });

    if(!containsArray(BC::Key::Custom::comm))
        setArray(BC::Key::Custom::comm, {
                    {{BC::Key::Custom::key,"devPath"},
                     {BC::Key::Custom::type,BC::Key::Custom::stringKey},
                     {BC::Key::Custom::label,"Device Path"}}
                 });

    save();
}

M4i2211x8::~M4i2211x8()
{

    if(p_handle != nullptr)
    {
        spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_STOP);
        spcm_dwInvalidateBuf(p_handle,SPCM_BUF_DATA);

        spcm_vClose(p_handle);
        p_handle = nullptr;
    }
}

void M4i2211x8::initialize()
{
    p_timer = new QTimer(this);
    connect(p_timer,&QTimer::timeout,this,&M4i2211x8::readWaveform);
}

bool M4i2211x8::testConnection()
{
    p_timer->stop();

    if(p_handle != nullptr)
    {
        stopCard();
        spcm_vClose(p_handle);
        p_handle = nullptr;
    }

    auto path = getArrayValue(BC::Key::Custom::comm,0,"devPath",QString("/dev/spcm0"));
    p_handle = spcm_hOpen(path.toLatin1().data());
    spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_RESET);


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


    if(errorCheck())
        return false;

    return true;
}

void M4i2211x8::readWaveform()
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
        stopCard();
        p_timer->stop();
        return;
    }

    if(stat & M2STAT_CARD_READY)
    {
        QVector<qint8> ba(d_bufferSize);
        spcm_dwDefTransfer_i64(p_handle,SPCM_BUF_DATA,SPCM_DIR_CARDTOPC,
                                   0,static_cast<void*>(ba.data()),
                                   0,static_cast<quint64>(d_bufferSize));
        spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA);
        errorCheck();
        emit waveformRead(ba);

        startCard();
    }

}

bool M4i2211x8::errorCheck()
{
    QByteArray errText(1000,'\0');
    if(spcm_dwGetErrorInfo_i32(p_handle,NULL,NULL,errText.data()) != ERR_OK)
    {
        stopCard();

        d_errorString = QString::fromLatin1(errText);
        emit hardwareFailure();
        emit logMessage(QString("An error occurred: %2").arg(d_errorString),LogHandler::Error);
        return true;
    }

    return false;
}

void M4i2211x8::startCard()
{
    spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER);

}

void M4i2211x8::stopCard()
{
    spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_STOP|M2CMD_DATA_STOPDMA);
    spcm_dwInvalidateBuf(p_handle,SPCM_BUF_DATA);

}

bool M4i2211x8::configure(const LifDigitizerConfig &c)
{

    static_cast<LifDigitizerConfig&>(*this) = c;
    d_channelOrder = LifDigitizerConfig::Interleaved;
    d_triggerChannel = 0;

    spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_RESET);

    spcm_dwSetParam_i32(p_handle,SPC_CARDMODE,SPC_REC_STD_SINGLE);

    if(d_refEnabled)
        spcm_dwSetParam_i32(p_handle,SPC_CHENABLE,CHANNEL0|CHANNEL1);
    else
        spcm_dwSetParam_i32(p_handle,SPC_CHENABLE,CHANNEL0);

    auto &ch = d_analogChannels[1];
    auto scale = ch.fullScale;
    if(scale < 0.35)
        ch.fullScale = 0.2;
    else if(scale < 0.75)
        ch.fullScale = 0.5;
    else if(scale < 1.75)
        ch.fullScale = 1.0;
    else
        ch.fullScale = 2.5;

    if(qAbs(ch.fullScale-scale)>0.01)
        emit logMessage(QString("LIF channel scale set to nearest allowed value (%1 V)").arg(ch.fullScale));
    spcm_dwSetParam_i32(p_handle,SPC_AMP0,static_cast<qint32>(round(ch.fullScale*1000.0)));

    //set offset to 0
    spcm_dwSetParam_i32(p_handle,SPC_OFFS0,0);
    ch.offset = 0.0;

    if(errorCheck())
        return false;

    if(d_refEnabled)
    {
        auto &ch2 = d_analogChannels[2];
        scale = ch2.fullScale;
        if(scale < 0.35)
            ch2.fullScale = 0.2;
        else if(scale < 0.75)
            ch2.fullScale = 0.5;
        else if(scale < 1.75)
            ch2.fullScale = 1.0;
        else
            ch2.fullScale = 2.5;

        if(qAbs(ch2.fullScale-scale)>0.01)
            emit logMessage(QString("Reference channel scale set to nearest allowed value (%1 V)").arg(ch2.fullScale));
        spcm_dwSetParam_i32(p_handle,SPC_AMP1,static_cast<qint32>(round(ch2.fullScale*1000.0)));

        //set offset to 0
        spcm_dwSetParam_i32(p_handle,SPC_OFFS1,0);
        ch2.offset = 0.0;

        if(errorCheck())
            return false;
    }

    //enforce constraint that record length must be a multiple of 32.
    int rl = (d_recordLength/32) * 32;
    rl = qMax(32,rl);
    rl = qMin(rl,65536);
    if(rl != d_recordLength)
        emit logMessage(QString("Record length set to %1 instead of %2 because it must be a multiple of 32 between 32 and 65536.").arg(rl).arg(d_recordLength),LogHandler::Warning);
    d_recordLength = rl;

    spcm_dwSetParam_i32(p_handle,SPC_CLOCKMODE,SPC_CM_INTPLL);
    spcm_dwSetParam_i64(p_handle,SPC_SAMPLERATE,static_cast<qint64>(d_sampleRate));

    d_bufferSize = d_recordLength;
    if(d_refEnabled)
        d_bufferSize *= 2;

    spcm_dwSetParam_i64(p_handle,SPC_MEMSIZE,static_cast<qint64>(d_bufferSize));
    spcm_dwSetParam_i64(p_handle,SPC_POSTTRIGGER,static_cast<qint64>(d_recordLength-32));

    if(errorCheck())
        return false;


    spcm_dwSetParam_i32(p_handle,SPC_TRIG_ORMASK,SPC_TMASK_NONE);
    spcm_dwSetParam_i32(p_handle,SPC_TRIG_ORMASK,SPC_TMASK_EXT0);
    spcm_dwSetParam_i32(p_handle,SPC_TRIG_EXT0_MODE,SPC_TM_POS);
    spcm_dwSetParam_i32(p_handle,SPC_TRIG_EXT0_LEVEL0,static_cast<int>(round(d_triggerLevel*1000.0)));
    spcm_dwSetParam_i32(p_handle,SPC_TRIG_DELAY,0);

    spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_WRITESETUP);

    return !errorCheck();
}

void M4i2211x8::beginAcquisition()
{
    startCard();
    p_timer->start(10);
}

void M4i2211x8::endAcquisition()
{
    stopCard();
    p_timer->stop();
}
