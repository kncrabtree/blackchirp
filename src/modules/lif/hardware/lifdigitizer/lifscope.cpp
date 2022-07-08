#include <modules/lif/hardware/lifdigitizer/lifscope.h>

LifScope::LifScope(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent, bool threaded, bool critical) :
    HardwareObject(BC::Key::LifDigi::lifScope,subKey,name,commType,parent,threaded,critical)
{
}

LifScope::~LifScope()
{

}

void LifScope::startConfigurationAcquisition(const LifDigitizerConfig &c)
{
    if(configure(c))
    {
        emit configAcqComplete(static_cast<LifDigitizerConfig>(*this));
        beginAcquisition();
    }
}
