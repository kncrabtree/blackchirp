#include "pico2206b.h"

#include <PicoStatus.h>
#include <ps2000aApi.h>
#include <math.h>
#include <QTimer>

Pico2206B::Pico2206B(QObject *parent) : MotorOscilloscope(parent)
{
    d_subKey = QString("pico2206b");
    d_prettyName = QString("Pico 2206B Oscilloscope");
    d_threaded = true;
    d_commType = CommunicationProtocol::Custom;

    d_handle = 0;
    connect(p_acquisitionTimer,&QTimer::timeout,this,&Pico2206B::endAcquisition);
    p_acquisitionTimer->setInterval(1);
}

Pico2206B::~Pico2206B()
{
    closeConnection();
}



bool Pico2206B::testConnection()
{
    closeConnection();

    status = ps2000aOpenUnit(&d_handle, NULL);
    if(status != PICO_OK)
    {
        emit hardwareFailure();
        emit logMessage(QString("Pico2206B opening failed."));
        return false;
    }
    return true;

    //do i need to call configure
}

void Pico2206B::initialize()
{
    p_acquisitionTimer->stop();
    testConnection();
}

bool Pico2206B::configure(const BlackChirp::MotorScopeConfig &sc)
{
    PS2000A_CHANNEL dataChannel, triggerChannel;

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
        range = PS2000A_20MV;
    else if (sc.verticalScale <= 0.05)
        range = PS2000A_50MV;
    else if (sc.verticalScale <= 0.1)
        range = PS2000A_100MV;
    else if (sc.verticalScale <= 0.2)
        range = PS2000A_200MV;
    else if (sc.verticalScale <= 0.5)
        range = PS2000A_500MV;
    else if (sc.verticalScale <= 1.0)
        range = PS2000A_1V;
    else if (sc.verticalScale <= 2.0)
        range = PS2000A_2V;
    else if (sc.verticalScale <= 5.0)
        range = PS2000A_5V;
    else if (sc.verticalScale <= 10.0)
        range = PS2000A_10V;
    else if (sc.verticalScale <= 20.0)
        range = PS2000A_20V;
    else
    {
        //what to do if the verticalScale is over 20V
    }

    status = ps2000aSetChannel(d_handle, dataChannel, true, PS2000A_DC, range, 0.0);
    if(status != PICO_OK)
    {
        emit hardwareFailure();
        emit logMessage(QString("Pico2206B channal setting failed."));
        return false;
    }

    float sampleInterval = 1.0 / sc.sampleRate;

    timebase = static_cast<uint32_t>(log2(500000000 * sampleInterval));
    noSamples = static_cast<int32_t>(sc.recordLength);

    float timeIntervalNanoseconds;
    status = ps2000aGetTimebase2(d_handle, timebase, noSamples, &timeIntervalNanoseconds, 0, NULL, 0);
    if(status != PICO_OK)
    {
        emit hardwareFailure();
        emit logMessage(QString("Pico2206B timebase setting failed."));
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

    int16_t threshold = 2^15;
    status = ps2000aSetSimpleTrigger(d_handle, 1, triggerChannel, threshold, direction, 0, 0);
    if(status != PICO_OK)
    {
        emit hardwareFailure();
        emit logMessage(QString("Pico2206B trigger setting failed."));
        return false;
    }


    status = ps2000aSetDataBuffer(d_handle, dataChannel, &d, noSamples, 0, PS2000A_RATIO_MODE_NONE);
    if(status != PICO_OK)
    {
        emit hardwareFailure();
        emit logMessage(QString("Pico2206B data buffer setting failed."));
        return false;
    }

    d_config = sc;
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

    status = ps2000aRunBlock(d_handle, 0, noSamples, timebase, 0, NULL, 0, NULL, NULL);
    if(status != PICO_OK)
    {
        emit hardwareFailure();
        emit logMessage(QString("Pico2206B data acquisition failed."));
        return;
    }
    p_acquisitionTimer->start();
}

void Pico2206B::endAcquisition()
{
    status = ps2000aIsReady(d_handle, &isReady);
    if(status != PICO_OK)
    {
        emit hardwareFailure();
        emit logMessage(QString("Pico2206B isReady function calling failed."));
        return;
    }
    if (isReady == 0)
        return;

    p_acquisitionTimer->stop();
    uint32_t noOfSamples;
    int16_t overflow;
    status = ps2000aGetValues(d_handle, 0, &noOfSamples, 1, PS2000A_RATIO_MODE_NONE, 0, &overflow);
    if(status != PICO_OK)
    {
        emit hardwareFailure();
        emit logMessage(QString("Pico2206B data passing failed."));
        return;
    }
    if(overflow != 0)
    {
        //some signal over range
    }

    emit traceAcquired(static_cast<QVector<double>>(d));
    d_acquiring = false;
    status = ps2000aStop(d_handle);
    if(status != PICO_OK)
    {
        emit hardwareFailure();
        emit logMessage(QString("Pico2206B stop failed."));
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

