#include "virtualpulsegenerator.h"

VirtualPulseGenerator::VirtualPulseGenerator(QObject *parent) :
    PulseGenerator(BC::Key::hwVirtual,BC::Key::vpGen,CommunicationProtocol::Virtual,8,parent)
{
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

QVariant VirtualPulseGenerator::read(const int index, const PulseGenConfig::Setting s)
{
    emit settingUpdate(index,s,d_config.setting(index,s));
    return d_config.setting(index,s);
}

double VirtualPulseGenerator::readRepRate()
{
    emit repRateUpdate(d_config.repRate());
    return d_config.repRate();
}

bool VirtualPulseGenerator::set(const int index, const PulseGenConfig::Setting s, const QVariant val)
{
    d_config.set(index,s,val);
    if(s == PulseGenConfig::RoleSetting)
    {
        auto r = val.value<PulseGenConfig::Role>();
        if(r != PulseGenConfig::NoRole)
        {
            d_config.set(index,PulseGenConfig::NameSetting,PulseGenConfig::roles.value(r));
            read(index,PulseGenConfig::NameSetting);
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
