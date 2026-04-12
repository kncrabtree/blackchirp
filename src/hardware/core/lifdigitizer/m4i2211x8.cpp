#include "m4i2211x8.h"
#include <hardware/core/hardwareregistration.h>

#include <QTimer>

using namespace Spectrum::M4i;

// Register hardware implementation
REGISTER_HARDWARE_META(M4i2211x8, "Spectrum M4i.2211-x8 LIF Digitizer")
REGISTER_HARDWARE_PROTOCOLS(M4i2211x8, CommunicationProtocol::Custom)
REGISTER_LIBRARY(M4i2211x8, SpectrumLibrary)
REGISTER_HARDWARE_SETTINGS(M4i2211x8,
    {BC::Key::Digi::numAnalogChannels, "Analog Channels", "Number of analog input channels", 2, 1, 128, HwSettingPriority::Required},
    {BC::Key::Digi::numDigitalChannels, "Digital Channels", "Number of digital input channels", 0, 0, 128, HwSettingPriority::Required},
    {BC::Key::Digi::hasAuxTriggerChannel, "Aux Trigger Channel", "Has auxiliary trigger input", true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::minFullScale, "Min Full Scale (V)", "Minimum full-scale voltage range", 0.05, QVariant{}, QVariant{}, HwSettingPriority::Important},
    {BC::Key::Digi::maxFullScale, "Max Full Scale (V)", "Maximum full-scale voltage range", 2.5, QVariant{}, QVariant{}, HwSettingPriority::Important},
    {BC::Key::Digi::minVOffset, "Min V Offset (V)", "Minimum vertical offset", -2.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxVOffset, "Max V Offset (V)", "Maximum vertical offset", 2.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::isTriggered, "Externally Triggered", "Digitizer uses external trigger signal", true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::minTrigDelay, "Min Trig Delay (us)", "Minimum trigger delay in microseconds", -10.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxTrigDelay, "Max Trig Delay (us)", "Maximum trigger delay in microseconds", 10.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::minTrigLevel, "Min Trig Level (V)", "Minimum trigger threshold voltage", -5.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxTrigLevel, "Max Trig Level (V)", "Maximum trigger threshold voltage", 5.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::canBlockAverage, "Block Average", "Supports block averaging mode", false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxBytes, "Max Bytes/Point", "Maximum bytes per sample", 1, 1, 4, HwSettingPriority::Optional},
    {BC::Key::Digi::maxRecordLength, "Max Record Length", "Maximum record length in samples", 1073741824, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxAverages, "Max Averages", "Maximum number of block averages", 10000, QVariant{}, QVariant{}, HwSettingPriority::Optional}
)
REGISTER_HARDWARE_ARRAY(M4i2211x8, BC::Key::Digi::sampleRates,
    "Sample Rates", "Available digitizer sample rates", HwSettingPriority::Important)
REGISTER_HARDWARE_ARRAY_ENTRY(M4i2211x8, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "78.125 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/32}})
REGISTER_HARDWARE_ARRAY_ENTRY(M4i2211x8, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "156.25 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/16}})
REGISTER_HARDWARE_ARRAY_ENTRY(M4i2211x8, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "312.5 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/8}})
REGISTER_HARDWARE_ARRAY_ENTRY(M4i2211x8, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "625 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/4}})
REGISTER_HARDWARE_ARRAY_ENTRY(M4i2211x8, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "1250 MSa/s"}, {BC::Key::Digi::srValue, 2.5e9/2}})

/*!
 * \brief Helper function to get SpectrumLibrary instance with availability check
 * \return Pointer to SpectrumLibrary if available, nullptr otherwise
 */
static SpectrumLibrary* getSpectrumLibrary()
{
    SpectrumLibrary& lib = SpectrumLibrary::instance();
    if (!lib.isAvailable()) {
        printf("Spectrum library not available: %s\n", lib.errorString().toLatin1().constData());
        return nullptr;
    }
    return &lib;
}

M4i2211x8::M4i2211x8(const QString& label, QObject *parent) :
    LifScope(QString(M4i2211x8::staticMetaObject.className()), label, parent),
    p_handle(nullptr)
{
    using namespace BC::Key::Digi;
    setDefault(canMultiRecord,false);
    setDefault(multiBlock,false);

    if(!containsArray(sampleRates))
        setArray(sampleRates,{
                     {{srText,"78.125 MSa/s"},{srValue,2.5e9/32}},
                     {{srText,"156.25 MSa/s"},{srValue,2.5e9/16}},
                     {{srText,"312.5 MSa/s"},{srValue,2.5e9/8}},
                     {{srText,"625 MSa/s"},{srValue,2.5e9/4}},
                     {{srText,"1250 MSa/s"},{srValue,2.5e9/2}},
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
        SpectrumLibrary* spcmLib = getSpectrumLibrary();
        if (spcmLib) {
            spcmLib->spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_STOP);
            spcmLib->spcm_dwInvalidateBuf(p_handle,SPCM_BUF_DATA);
            spcmLib->spcm_vClose(p_handle);
        }
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
    SpectrumLibrary* spcmLib = getSpectrumLibrary();
    if (!spcmLib) {
        d_errorString = "Spectrum library not available";
        return false;
    }
    
    p_timer->stop();

    if(p_handle != nullptr)
    {
        stopCard();
        spcmLib->spcm_vClose(p_handle);
        p_handle = nullptr;
    }

    auto path = getArrayValue(BC::Key::Custom::comm,0,"devPath",QString("/dev/spcm0"));
    p_handle = spcmLib->spcm_hOpen(path.toLatin1().data());
    spcmLib->spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_RESET);


    if(p_handle == nullptr)
    {
        d_errorString = QString("Could not connect to digitizer. Verify that %1 exists and is accessible.").arg(QString(path));
        return false;
    }

    qint32 cType = 0;
    spcmLib->spcm_dwGetParam_i32(p_handle,SPC_PCITYP,&cType);

    qint32 serialNo = 0;
    spcmLib->spcm_dwGetParam_i32(p_handle,SPC_PCISERIALNO,&serialNo);

    qint32 driVer = 0;
    spcmLib->spcm_dwGetParam_i32(p_handle,SPC_GETDRVVERSION,&driVer);

    qint32 kerVer = 0;
    spcmLib->spcm_dwGetParam_i32(p_handle,SPC_GETKERNELVERSION,&kerVer);

    QByteArray errText(1000,'\0');
    if(spcmLib->spcm_dwGetErrorInfo_i32(p_handle,NULL,NULL,errText.data()) != ERR_OK)
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
    SpectrumLibrary* spcmLib = getSpectrumLibrary();
    if (!spcmLib) {
        emit logMessage("Spectrum library not available", LogHandler::Error);
        emit hardwareFailure();
        stopCard();
        p_timer->stop();
        return;
    }
    
    //check to see if a data block is ready
    qint32 stat = 0;
    spcmLib->spcm_dwGetParam_i32(p_handle,SPC_M2STATUS,&stat);
    if(stat & M2STAT_DATA_ERROR) //internal error
    {
        QByteArray errText(1000,'\0');
        if(spcmLib->spcm_dwGetErrorInfo_i32(p_handle,NULL,NULL,errText.data()) != ERR_OK)
            emit logMessage(QString::fromLatin1(errText),LogHandler::Error);

        emit hardwareFailure();
        stopCard();
        p_timer->stop();
        return;
    }

    if(stat & M2STAT_CARD_READY)
    {
        QVector<qint8> ba(d_bufferSize);
        spcmLib->spcm_dwDefTransfer_i64(p_handle,SPCM_BUF_DATA,SPCM_DIR_CARDTOPC,
                                   0,static_cast<void*>(ba.data()),
                                   0,static_cast<quint64>(d_bufferSize));
        spcmLib->spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA);
        errorCheck();
        emitWaveform(ba);

        startCard();
    }

}

bool M4i2211x8::errorCheck()
{
    SpectrumLibrary* spcmLib = getSpectrumLibrary();
    if (!spcmLib) {
        d_errorString = "Spectrum library not available";
        emit hardwareFailure();
        emit logMessage(QString("An error occurred: %1").arg(d_errorString),LogHandler::Error);
        return true;
    }
    
    QByteArray errText(1000,'\0');
    if(spcmLib->spcm_dwGetErrorInfo_i32(p_handle,NULL,NULL,errText.data()) != ERR_OK)
    {
        stopCard();

        d_errorString = QString::fromLatin1(errText);
        emit hardwareFailure();
        emit logMessage(QString("An error occurred: %1").arg(d_errorString),LogHandler::Error);
        return true;
    }

    return false;
}

void M4i2211x8::startCard()
{
    SpectrumLibrary* spcmLib = getSpectrumLibrary();
    if (spcmLib) {
        spcmLib->spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER);
    }
}

void M4i2211x8::stopCard()
{
    SpectrumLibrary* spcmLib = getSpectrumLibrary();
    if (spcmLib) {
        spcmLib->spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_STOP|M2CMD_DATA_STOPDMA);
        spcmLib->spcm_dwInvalidateBuf(p_handle,SPCM_BUF_DATA);
    }
}

bool M4i2211x8::configure(const LifDigitizerConfig &c)
{
    SpectrumLibrary* spcmLib = getSpectrumLibrary();
    if (!spcmLib) {
        emit logMessage("Spectrum library not available", LogHandler::Error);
        return false;
    }

    static_cast<LifDigitizerConfig&>(*this) = c;
    d_channelOrder = LifDigitizerConfig::Interleaved;
    d_triggerChannel = 0;

    spcmLib->spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_RESET);

    spcmLib->spcm_dwSetParam_i32(p_handle,SPC_CARDMODE,SPC_REC_STD_SINGLE);

    if(d_refEnabled)
        spcmLib->spcm_dwSetParam_i32(p_handle,SPC_CHENABLE,CHANNEL0|CHANNEL1);
    else
        spcmLib->spcm_dwSetParam_i32(p_handle,SPC_CHENABLE,CHANNEL0);

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
    spcmLib->spcm_dwSetParam_i32(p_handle,SPC_AMP0,static_cast<qint32>(round(ch.fullScale*1000.0)));

    //set offset to 0
    spcmLib->spcm_dwSetParam_i32(p_handle,SPC_OFFS0,0);
    ch.offset = 0.0;

    spcmLib->spcm_dwSetParam_i32(p_handle,SPC_ACDC0,0);

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
        spcmLib->spcm_dwSetParam_i32(p_handle,SPC_AMP1,static_cast<qint32>(round(ch2.fullScale*1000.0)));

        //set offset to 0
        spcmLib->spcm_dwSetParam_i32(p_handle,SPC_OFFS1,0);
        ch2.offset = 0.0;

        spcmLib->spcm_dwSetParam_i32(p_handle,SPC_ACDC1,0);

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

    spcmLib->spcm_dwSetParam_i32(p_handle,SPC_CLOCKMODE,SPC_CM_INTPLL);
    spcmLib->spcm_dwSetParam_i64(p_handle,SPC_SAMPLERATE,static_cast<qint64>(d_sampleRate));

    d_bufferSize = d_recordLength;
    if(d_refEnabled)
        d_bufferSize *= 2;

    spcmLib->spcm_dwSetParam_i64(p_handle,SPC_MEMSIZE,static_cast<qint64>(d_recordLength));
    spcmLib->spcm_dwSetParam_i64(p_handle,SPC_POSTTRIGGER,static_cast<qint64>(d_recordLength-32));

    if(errorCheck())
        return false;


    spcmLib->spcm_dwSetParam_i32(p_handle,SPC_TRIG_ORMASK,SPC_TMASK_NONE);
    spcmLib->spcm_dwSetParam_i32(p_handle,SPC_TRIG_ORMASK,SPC_TMASK_EXT0);
    spcmLib->spcm_dwSetParam_i32(p_handle,SPC_TRIG_EXT0_MODE,SPC_TM_POS);
    spcmLib->spcm_dwSetParam_i32(p_handle,SPC_TRIG_EXT0_LEVEL0,static_cast<int>(round(d_triggerLevel*1000.0)));
    spcmLib->spcm_dwSetParam_i32(p_handle,SPC_TRIG_DELAY,0);

    spcmLib->spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_WRITESETUP);

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
