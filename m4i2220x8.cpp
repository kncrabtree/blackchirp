#include "m4i2220x8.h"

M4i2220x8::M4i2220x8(QObject *parent) : FtmwScope(parent), p_handle(nullptr)
{
    d_subKey = QString("m4i2220x8");
    d_prettyName = QString("Spectrum Instrumentation M4i.2220-x8 Digitizer");
    d_commType = CommunicationProtocol::Custom;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);
    double bandwidth = s.value(QString("bandwidth"),1250.0).toDouble();
    s.setValue(QString("bandwidth"),bandwidth);
    s.endGroup();
    s.endGroup();
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
    if(p_handle == nullptr)
    {
        p_handle = spcm_hOpen(QByteArray("/dev/spcm0").data());
    }

    if(p_handle == nullptr)
    {
        emit connected(false,QString("Could not connect to digitizer."));
        return false;
    }
    emit connected();
    return true;
}

void M4i2220x8::initialize()
{
    testConnection();
}

Experiment M4i2220x8::prepareForExperiment(Experiment exp)
{
    return exp;
}

void M4i2220x8::beginAcquisition()
{
}

void M4i2220x8::endAcquisition()
{
}

void M4i2220x8::readTimeData()
{
}

void M4i2220x8::readWaveform()
{
}
