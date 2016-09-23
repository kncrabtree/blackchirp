#include "pico2206b.h"

#include <PicoStatus.h>
#include <ps2000aApi.h>
#include <QtCore/qmath.h>
#include <QTimer>

Pico2206B::Pico2206B(QObject *parent) : MotorOscilloscope(parent)
{
    d_subKey = QString("pico2206b");
    d_prettyName = QString("Pico 2206B Oscilloscope");
    d_threaded = true;
    d_commType = CommunicationProtocol::Custom;

    d_handle = 0;

}

Pico2206B::~Pico2206B()
{
    closeConnection();
}



bool Pico2206B::testConnection()
{
    p_acquisitionTimer->stop();
    closeConnection();

    status = ps2000aOpenUnit(&d_handle, NULL);
    if(status != PICO_OK)
    {
        emit connected(false);
        emit logMessage(QString("Pico2206B opening failed. Error code: %1").arg(status));
        return false;
    }

    emit connected();

    configure(d_config);

    return true;

}

void Pico2206B::initialize()
{
    p_acquisitionTimer = new QTimer(this);
    connect(p_acquisitionTimer,&QTimer::timeout,this,&Pico2206B::endAcquisition);
    p_acquisitionTimer->setInterval(1000);

    d_acquiring = false;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    d_config.dataChannel = s.value(QString("dataChannel"),1).toInt();
    d_config.verticalScale = s.value(QString("verticalScale"),5.0).toDouble();
    d_config.recordLength = s.value(QString("sampleRate"),100).toInt();
    d_config.sampleRate = s.value(QString("sampleRate"),500.0).toDouble();
    d_config.triggerChannel = s.value(QString("triggerChannel"),2).toInt();
    d_config.slope = static_cast<BlackChirp::ScopeTriggerSlope>(s.value(QString("slope"),BlackChirp::ScopeTriggerSlope::RisingEdge).toUInt());
    //d_config.byteOrder = static_cast<QDataStream::ByteOrder>(s.value(QString("byteOrder"),QDataStream::ByteOrder::BigEndian).toUInt());
    //d_config.bytesPerPoint = s.value(QString("bytesPerPoing"),100).toInt();
    s.endGroup();
    s.endGroup();

    testConnection();


}

bool Pico2206B::configure(const BlackChirp::MotorScopeConfig &sc)
{
    d_config = sc;

    PS2000A_CHANNEL dataChannel, triggerChannel;

    //Do I need to judge recordlenth times interval < 0.2

    if (sc.dataChannel == 1)
        dataChannel = PS2000A_CHANNEL_A;
    else
        dataChannel = PS2000A_CHANNEL_B;

    if (sc.triggerChannel == 1)
        triggerChannel = PS2000A_CHANNEL_A;
    else
        triggerChannel = PS2000A_CHANNEL_B;

    PS2000A_RANGE range;
    if (sc.verticalScale <= 0.02)
    {
        d_config.verticalScale = 0.02;
        range = PS2000A_20MV;
    }
    else if (sc.verticalScale <= 0.05)
    {
        d_config.verticalScale = 0.05;
        range = PS2000A_50MV;
    }
    else if (sc.verticalScale <= 0.1)
    {
        d_config.verticalScale = 0.1;
        range = PS2000A_100MV;
    }
    else if (sc.verticalScale <= 0.2)
    {
        d_config.verticalScale = 0.2;
        range = PS2000A_200MV;
    }
    else if (sc.verticalScale <= 0.5)
    {
        d_config.verticalScale = 0.5;
        range = PS2000A_500MV;
    }
    else if (sc.verticalScale <= 1.0)
    {
        d_config.verticalScale = 1.0;
        range = PS2000A_1V;
    }
    else if (sc.verticalScale <= 2.0)
    {
        d_config.verticalScale = 2.0;
        range = PS2000A_2V;
    }
    else if (sc.verticalScale <= 5.0)
    {
        d_config.verticalScale = 5.0;
        range = PS2000A_5V;
    }
    else if (sc.verticalScale <= 10.0)
    {
        d_config.verticalScale = 10.0;
        range = PS2000A_10V;
    }
    else //if (sc.verticalScale <= 20.0)
    {
        d_config.verticalScale = 20.0;
        range = PS2000A_20V;
    }
    //else
    //{
        //what to do if the verticalScale is over 20V
    //}

    status = ps2000aSetChannel(d_handle, dataChannel, true, PS2000A_DC, range, 0.0);
    if(status != PICO_OK)
    {
        emit connected(false);
        emit logMessage(QString("Pico2206B channal setting failed. Error code: %1").arg(status));
        return false;
    }

    double sampleInterval = (sc.sampleRate * qPow(10,-9));
    timebase = 62500000 * sampleInterval + 2;

    noSamples = sc.recordLength;

    status = ps2000aGetTimebase(d_handle, timebase, noSamples, NULL, 0, NULL, 0);
    if(status != PICO_OK)
    {
        emit connected(false);
        emit logMessage(QString("Pico2206B timebase setting failed. Error code: %1").arg(status));
        return false;
    }

    PS2000A_THRESHOLD_DIRECTION direction;
    switch(sc.slope)
    {
    case BlackChirp::FallingEdge:
        direction = enPS2000AThresholdDirection::PS2000A_FALLING;
        break;
    case BlackChirp::RisingEdge:
        direction = enPS2000AThresholdDirection::PS2000A_RISING;
        break;
    }

    int16_t threshold = qPow(2,13);
    status = ps2000aSetSimpleTrigger(d_handle, 1, triggerChannel, threshold, direction, 0, 0);
    if(status != PICO_OK)
    {
        emit connected(false);
        emit logMessage(QString("Pico2206B trigger setting failed. Error code: %1").arg(status));
        return false;
    }

    d_buffer.resize(noSamples);
    status = ps2000aSetDataBuffer(d_handle, dataChannel, d_buffer.data(), noSamples, 0, PS2000A_RATIO_MODE_NONE);
    if(status != PICO_OK)
    {
        emit connected(false);
        emit logMessage(QString("Pico2206B data buffer setting failed. Error code: %1").arg(status));
        return false;
    }

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    s.setValue(QString("dataChannel"),sc.dataChannel);
    s.setValue(QString("verticalScale"),sc.verticalScale);
    s.setValue(QString("recordlength"),sc.recordLength);
    s.setValue(QString("sampleRate"),sc.sampleRate);
    s.setValue(QString("triggerChannel"),sc.triggerChannel);
    s.setValue(QString("slope"),static_cast<uint>(sc.slope));
    //s.setValue(QString("byteOrder"),static_cast<uint>(sc.byteOrder));
    //s.setValue(QString("bytesPerPoint"),sc.bytesPerPoint);
    s.endGroup();
    s.endGroup();

    beginAcquisition();

    return true;
}

MotorScan Pico2206B::prepareForMotorScan(MotorScan s)
{
    bool ok = configure(s.scopeConfig());
    if(!ok)
        s.setHardwareError();
    else
        s.setScopeConfig(d_config);

    return s;
}

void Pico2206B::beginAcquisition()
{
    if (d_acquiring == true)
        return;

    emit logMessage(QString("start acqu funtion"));

    status = ps2000aRunBlock(d_handle, 0, noSamples, timebase, 0, NULL, 0, NULL, NULL);
    if(status != PICO_OK)
    {
        emit connected(false);
        emit logMessage(QString("Pico2206B data acquisition failed. Error code: %1").arg(status));
        return;
    }
    p_acquisitionTimer->start();

    emit logMessage(QString("start acqu process"));
}

void Pico2206B::endAcquisition()
{
    emit logMessage(QString("start end acqu function"));
    status = ps2000aIsReady(d_handle, &isReady);
    if(status != PICO_OK)
    {
        emit connected(false);
        emit logMessage(QString("Pico2206B isReady function calling failed. Error code: %1").arg(status));
        return;
    }
    if (isReady == 0)
        return;

    emit logMessage(QString("got all data"));
    p_acquisitionTimer->stop();

    uint32_t noOfSamples = noSamples;
    int16_t overflow = 1;
    status = ps2000aGetValues(d_handle, 0, &noOfSamples, 1, PS2000A_RATIO_MODE_NONE, 0, &overflow);
    if(status != PICO_OK)
    {
        emit connected(false);
        emit logMessage(QString("Pico2206B data passing failed. Error code: %1").arg(status));
        return;
    }
    if(overflow != 0)
    {
        //some signal over range
    }

    QVector<double> d;
    d.resize(noSamples);

    for (int i = 0; i < noSamples; ++i)
    {
        d[i] = d_buffer[i] / 32512.0 * d_config.verticalScale;
    }

    emit traceAcquired(d);

    d_acquiring = false;
    status = ps2000aStop(d_handle);
    if(status != PICO_OK)
    {
        emit connected(false);
        emit logMessage(QString("Pico2206B stop failed. Error code: %1").arg(status));
        return;
    }
    beginAcquisition();
}


void Pico2206B::closeConnection()
{
    if(d_handle)
        ps2000aCloseUnit(d_handle);

    d_handle = 0;
}

