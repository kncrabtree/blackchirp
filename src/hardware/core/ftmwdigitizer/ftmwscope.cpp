#include <hardware/core/ftmwdigitizer/ftmwscope.h>

FtmwScope::FtmwScope(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent, bool threaded, bool critical) :
    HardwareObject(BC::Key::FtmwScope::ftmwScope,subKey,name,commType,parent,threaded,critical)
{

}

FtmwScope::~FtmwScope()
{

}





QStringList FtmwScope::forbiddenKeys() const
{
    return {BC::Key::Digi::numAnalogChannels, BC::Key::Digi::numDigitalChannels};
}
