#include "m4i2211x8.h"

#include <QTimer>

M4i2211x8::M4i2211x8(QObject *parent) : LifScope (parent), p_handle(nullptr), p_m4iBuffer(nullptr), d_timerInterval(50), d_running(false)
{
    d_subKey = QString("m4i2211x8");
    d_prettyName = QString("Spectrum Instrumentation M4i.2211-x8 Digitizer");
    d_commType = CommunicationProtocol::Custom;
    d_threaded = true;

    //load settings from last use, if possible
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    d_config.vScale1 = s.value(QString("vScale"),0.2).toDouble();
    d_config.vScale2 = s.value(QString("vScaleRef"),0.2).toDouble();
    d_config.sampleRate = s.value(QString("sampleRate"),1.25e9).toDouble();
    d_config.recordLength = s.value(QString("numSamples"),8192).toInt();
    d_config.refEnabled = s.value(QString("refEnabled"),false).toBool();
    s.endGroup();
    s.endGroup();

    d_config.bytesPerPoint = 1;
    d_config.byteOrder = QDataStream::LittleEndian;
    d_config.yMult1 = d_config.vScale1/128.0;
    d_config.yMult2 = d_config.vScale2/128.0;
    d_config.xIncr = 1.0/d_config.sampleRate;

    d_bufferSize = d_config.recordLength;
    if(d_config.refEnabled)
        d_bufferSize *= 2;
    p_m4iBuffer = new char[d_bufferSize];
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

    if(p_m4iBuffer != nullptr)
    {
        delete[] p_m4iBuffer;
        p_m4iBuffer = nullptr;
    }
}


void M4i2211x8::readSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);


    double bandwidth = s.value(QString("bandwidth"),500.0).toDouble();
    s.setValue(QString("bandwidth"),bandwidth);
    d_timerInterval = s.value(QString("timerIntervalMs"),50).toInt();
    s.setValue(QString("timerIntervalMs"),d_timerInterval);

    s.setValue(QString("minVScale"),0.2);
    s.setValue(QString("maxVScale"),2.5);

    if(s.beginReadArray(QString("sampleRates")) < 1)
    {
        s.endArray();

        QList<QPair<QString,double>> sampleRates;
        for(int i=0; i<6; i++)
        {
            QString txt = QString("%1 MSa/S").arg(round(1.25e3/( 1 << i )),4);
            double val = 1.25e9/(static_cast<double>( 1 << i));
            sampleRates << qMakePair(txt,val);
        }

        s.beginWriteArray(QString("sampleRates"));
        for(int i=0; i<sampleRates.size(); i++)
        {
            s.setArrayIndex(i);
            s.setValue(QString("text"),sampleRates.at(i).first);
            s.setValue(QString("val"),sampleRates.at(i).second);
        }
    }

    s.endArray();
    s.beginWriteArray(QString("comm"));
    s.setArrayIndex(0);
    s.setValue(QString("name"),QString("Device Path"));
    s.setValue(QString("key"),QString("devPath"));
    s.setValue(QString("type"),QString("string"));
    s.endArray();

    s.endGroup();
    s.endGroup();
}

void M4i2211x8::initialize()
{
    p_timer = new QTimer(this);
    connect(p_timer,&QTimer::timeout,this,&M4i2211x8::queryScope);
}

bool M4i2211x8::testConnection()
{
    p_timer->stop();

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    QByteArray path = s.value(QString("devPath"),QString("/dev/spcm0")).toString().toLatin1();
    s.endGroup();
    s.endGroup();

    if(p_handle != nullptr)
    {
        stopCard();
        spcm_vClose(p_handle);
        p_handle = nullptr;
    }

    p_handle = spcm_hOpen(path.data());
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

    //try to restore last used parameters
    blockSignals(true);
    setRefEnabled(d_config.refEnabled); //this will also set the ref V scale!
    setLifVScale(d_config.vScale1);
    setHorizontalConfig(d_config.sampleRate,d_config.recordLength);
    blockSignals(false);

    //configure type of acquisition etc here
    spcm_dwSetParam_i32(p_handle,SPC_CLOCKMODE,SPC_CM_INTPLL);
    spcm_dwSetParam_i32(p_handle,SPC_TRIG_ORMASK,SPC_TMASK_NONE);
    spcm_dwSetParam_i32(p_handle,SPC_TRIG_ORMASK,SPC_TMASK_EXT0);
    spcm_dwSetParam_i32(p_handle,SPC_TRIG_EXT0_MODE,SPC_TM_POS);
    spcm_dwSetParam_i32(p_handle,SPC_TRIG_EXT0_LEVEL0,2200);
    spcm_dwSetParam_i32(p_handle,SPC_CARDMODE,SPC_REC_STD_SINGLE);


    if(errorCheck())
        return false;

    emit configUpdated(d_config);

    startCard();
    p_timer->start(d_timerInterval);

    return true;
}

void M4i2211x8::setLifVScale(double scale)
{

    bool wasRunning = stopCard();

    if(scale < 0.35)
        d_config.vScale1 = 0.2;
    else if(scale < 0.75)
        d_config.vScale1 = 0.5;
    else if(scale < 1.75)
        d_config.vScale1 = 1.0;
    else
        d_config.vScale1 = 2.5;

    spcm_dwSetParam_i32(p_handle,SPC_AMP0,static_cast<qint32>(round(d_config.vScale1*1000.0)));
    d_config.yMult1 = d_config.vScale1/128.0;

    //set offset to 0
    spcm_dwSetParam_i32(p_handle,SPC_OFFS0,0);

    if(errorCheck())
        return;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    s.setValue(QString("vScale"),d_config.vScale1);
    s.endGroup();
    s.endGroup();

    emit configUpdated(d_config);

    if(wasRunning)
        startCard();

}

void M4i2211x8::setRefVScale(double scale)
{

    bool wasRunning = stopCard();

    if(scale < 0.35)
        d_config.vScale2 = 0.2;
    else if(scale < 0.75)
        d_config.vScale2 = 0.5;
    else if(scale < 1.75)
        d_config.vScale2 = 1.0;
    else
        d_config.vScale2 = 2.5;

    spcm_dwSetParam_i32(p_handle,SPC_AMP1,static_cast<qint32>(round(d_config.vScale2*1000.0)));
    d_config.yMult2 = d_config.vScale2/128.0;

    //set offset to 0
    spcm_dwSetParam_i32(p_handle,SPC_OFFS1,0);

    if(errorCheck())
        return;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    s.setValue(QString("vScaleRef"),d_config.vScale2);
    s.endGroup();
    s.endGroup();
    emit configUpdated(d_config);

    if(wasRunning)
        startCard();
}

void M4i2211x8::setHorizontalConfig(double sampleRate, int recLen)
{
    bool wasRunning = stopCard();

    d_config.recordLength = recLen;
    d_config.sampleRate = sampleRate;

    spcm_dwSetParam_i64(p_handle,SPC_SAMPLERATE,static_cast<qint64>(sampleRate));

    configureMemory();

    if(errorCheck())
        return;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    s.setValue(QString("sampleRate"),d_config.sampleRate);
    s.setValue(QString("numSamples"),d_config.recordLength);
    s.endGroup();
    s.endGroup();
    emit configUpdated(d_config);

    if(wasRunning)
        startCard();
}

void M4i2211x8::setRefEnabled(bool en)
{
    bool wasRunning = stopCard();

    d_config.refEnabled = en;

    if(en)
        spcm_dwSetParam_i32(p_handle,SPC_CHENABLE,CHANNEL0|CHANNEL1);
    else
        spcm_dwSetParam_i32(p_handle,SPC_CHENABLE,CHANNEL0);

    configureMemory();

    if(en)
        setRefVScale(d_config.vScale2);

    if(errorCheck())
        return;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    s.setValue(QString("refEnabled"),d_config.refEnabled);
    s.endGroup();
    s.endGroup();

    emit configUpdated(d_config);

    if(wasRunning)
        startCard();

}

void M4i2211x8::queryScope()
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
        stopCard();
        p_timer->stop();
        return;
    }

    if(stat & M2STAT_CARD_READY)
    {
        spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA);
        emit waveformRead(LifTrace(d_config,QByteArray::fromRawData(p_m4iBuffer,d_bufferSize)));

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
        emit logMessage(QString("An error occurred: %2").arg(d_errorString),BlackChirp::LogError);
        if(p_m4iBuffer != nullptr)
        {
            delete[] p_m4iBuffer;
            p_m4iBuffer = nullptr;
        }
        return true;
    }

    return false;
}

void M4i2211x8::configureMemory()
{
    bool wasRunning = stopCard();

    d_bufferSize = d_config.recordLength;
    if(d_config.refEnabled)
        d_bufferSize *= 2;

    spcm_dwInvalidateBuf(p_handle,SPCM_BUF_DATA);
    spcm_dwSetParam_i64(p_handle,SPC_MEMSIZE,static_cast<qint64>(d_bufferSize));
    spcm_dwSetParam_i64(p_handle,SPC_POSTTRIGGER,static_cast<qint64>(d_config.recordLength-32));


    if(p_m4iBuffer != nullptr)
    {
        delete [] p_m4iBuffer;
        p_m4iBuffer = nullptr;
    }

    p_m4iBuffer = new char[d_bufferSize];

    if(wasRunning)
        startCard();
}

void M4i2211x8::startCard()
{
    spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER);
    spcm_dwDefTransfer_i64(p_handle,SPCM_BUF_DATA,SPCM_DIR_CARDTOPC,0,static_cast<void*>(p_m4iBuffer),0,static_cast<quint64>(d_bufferSize));
    d_running = true;

}

bool M4i2211x8::stopCard()
{
    bool out = d_running;
    spcm_dwSetParam_i32(p_handle,SPC_M2CMD,M2CMD_CARD_STOP|M2CMD_DATA_STOPDMA);
    spcm_dwInvalidateBuf(p_handle,SPCM_BUF_DATA);
    d_running = false;

    return out;
}
