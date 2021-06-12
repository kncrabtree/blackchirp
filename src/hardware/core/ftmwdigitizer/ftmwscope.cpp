#include <src/hardware/core/ftmwdigitizer/ftmwscope.h>

FtmwScope::FtmwScope(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent, bool threaded, bool critical) :
    HardwareObject(BC::Key::ftmwScope,subKey,name,commType,parent,threaded,critical)
{

}

FtmwScope::~FtmwScope()
{

}



