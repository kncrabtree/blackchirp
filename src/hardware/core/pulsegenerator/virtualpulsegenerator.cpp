#include "virtualpulsegenerator.h"

VirtualPulseGenerator::VirtualPulseGenerator(QObject *parent) : PulseGenerator(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual Pulse Generator");
    d_commType = CommunicationProtocol::Virtual;
    d_threaded = false;
    d_numChannels = 8;
}

VirtualPulseGenerator::~VirtualPulseGenerator()
{

}

bool VirtualPulseGenerator::testConnection()
{
    blockSignals(true);
    readAll();
    blockSignals(false);

    emit configUpdate(d_config);
    return true;
}

QVariant VirtualPulseGenerator::read(const int index, const BlackChirp::PulseSetting s)
{
    emit settingUpdate(index,s,d_config.setting(index,s));
    return d_config.setting(index,s);
}

double VirtualPulseGenerator::readRepRate()
{
    emit repRateUpdate(d_config.repRate());
    return d_config.repRate();
}

bool VirtualPulseGenerator::set(const int index, const BlackChirp::PulseSetting s, const QVariant val)
{
    d_config.set(index,s,val);
    if(s == BlackChirp::PulseRoleSetting)
    {
        if(static_cast<BlackChirp::PulseRole>(val.toInt()) != BlackChirp::NoPulseRole)
        {
            d_config.set(index,BlackChirp::PulseNameSetting,BlackChirp::getPulseName(static_cast<BlackChirp::PulseRole>(val.toInt())));
            read(index,BlackChirp::PulseNameSetting);
        }
    }
    read(index,s);
    return true;
}

bool VirtualPulseGenerator::setRepRate(double d)
{
    d_config.setRepRate(d);
    readRepRate();
    return true;
}
