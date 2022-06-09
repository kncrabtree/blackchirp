#include <modules/lif/hardware/lifdigitizer/lifscope.h>

LifScope::LifScope(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent, bool threaded, bool critical) :
    HardwareObject(BC::Key::lifScope,subKey,name,commType,parent,threaded,critical)
{
}

LifScope::~LifScope()
{

}
