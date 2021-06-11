#include <src/hardware/core/chirpsource/awg.h>

AWG::AWG(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent, bool threaded, bool critical) :
    HardwareObject(BC::Key::awg,subKey,name,commType,parent,threaded,critical)
{

}

AWG::~AWG()
{

}
