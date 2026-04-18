#include <hardware/core/communication/communicationprotocol.h>
#include <hardware/core/hardwareobject.h>
#include <data/settings/hardwarekeys.h>
#include <QCoreApplication>

CommunicationProtocol::CommunicationProtocol(QString key, QObject *parent) :
    QObject(parent),d_key(key)
{

}

CommunicationProtocol::~CommunicationProtocol()
{

}

bool CommunicationProtocol::writeCmd(const QString &cmd)
{
    if(_device() == nullptr)
        return true;

    if(!_device()->isOpen())
    {
        emit hardwareFailure();
        bcError("Could not write command. Serial port is not open."_L1);
        bcDebug(u"%1 writeCmd: Could not write command. Serial port is not open. Command = %2"_s.arg(d_key, cmd));
        return false;
    }

    qint64 ret = _device()->write(cmd.toLatin1());

    if(ret == -1)
    {
        emit hardwareFailure();
        bcError("Could not write command."_L1);
        bcDebug(u"%1 writeCmd: Could not write command. Command = %2"_s.arg(d_key, cmd));
        return false;
    }
    return true;
}

bool CommunicationProtocol::writeBinary(const QByteArray &dat)
{
    if(_device() == nullptr)
        return true;

    if(!_device()->isOpen())
    {
        emit hardwareFailure();
        bcError("Could not write binary data. Serial port is not open."_L1);
        bcDebug(u"%1 writeBinary: Could not write binary data. Serial port is not open. Data hex = %2"_s.arg(d_key, QString(dat.toHex())));
        return false;
    }

    qint64 ret = _device()->write(dat);

    if(ret == -1)
    {
        emit hardwareFailure();
        bcError("Could not write binary data."_L1);
        bcDebug(u"%1 writeBinary: Could not write binary data. Data hex = %2"_s.arg(d_key, QString(dat.toHex())));
        return false;
    }
    return true;
}

QByteArray CommunicationProtocol::queryCmd(const QString &cmd, bool suppressError)
{
    if(_device() == nullptr)
        return QByteArray();

    if(_device()->bytesAvailable())
        _device()->readAll();

    qint64 ret = _device()->write(cmd.toLatin1());

    if(ret == -1)
    {
        if(!suppressError)
        {
            emit hardwareFailure();
            bcError("Could not write query."_L1);
            bcDebug(u"%1 queryCmd: Could not write query. Query = %2"_s.arg(d_key, cmd));
        }
        return QByteArray();
    }

    while (_device()->bytesToWrite() > 0) {
        if(!_device()->waitForBytesWritten(30000))
        {
            if(!suppressError)
            {
                emit hardwareFailure();
                bcError("Timed out while waiting for query write."_L1);
                bcDebug(u"%1 queryCmd: Timed out while waiting for query write. Query = %2"_s.arg(d_key, cmd));
            }
            return QByteArray();
        }
    }

    //write, return response
    if(!d_useTermChar || d_readTerminator.isEmpty())
    {
        if(!_device()->waitForReadyRead(d_timeOut))
        {
            if(!suppressError)
            {
                emit hardwareFailure();
                bcError("Did not respond to query."_L1);
                bcDebug(u"%1 queryCmd: Did not respond to query. Query = %2"_s.arg(d_key, cmd));
            }
            return QByteArray();
        }

        return _device()->readAll();
    }
    else
    {
        QByteArray out;
        bool done = false;
        while(!done)
        {
            if(!_device()->waitForReadyRead(d_timeOut))
                break;

            QByteArray t = _device()->readAll();
            if(t.contains(d_readTerminator))
            {
                out.append(t.mid(0,t.indexOf(d_readTerminator)));
                return out;
            }
            else
            {
                out.append(t);
                if(out.contains(d_readTerminator)) //this handles cases in which there is a multi-character read terminator
                    return out.mid(0,out.indexOf(d_readTerminator));
            }
        }

        if(!suppressError)
        {
            emit hardwareFailure();
            bcError("Timed out while waiting for termination character."_L1);
            bcDebug(u"%1 queryCmd: Query = %2, partial response = %3, hex = %4"_s.arg(d_key, cmd, QString(out), QString(out.toHex())));
        }
        return out;
    }
}

QByteArray CommunicationProtocol::readBytes(qint64 n, bool suppressError)
{
    if(_device() == nullptr)
        return QByteArray();

    if(n<1)
        return QByteArray();

    auto ba = _device()->bytesAvailable();


    while(ba < n)
    {
        if(!_device()->waitForReadyRead(d_timeOut))
        {
            if(!suppressError)
            {
                emit hardwareFailure();
                bcError(u"Could not read %1 bytes; timeout error."_s.arg(n));
            }
            return {};
        }
        else
            ba = _device()->bytesAvailable();
    }

    return _device()->read(n);
}

void CommunicationProtocol::setErrorString(const QString str)
{
    d_errorString = str;
}

QString CommunicationProtocol::errorString()
{
    QString out = d_errorString;
    d_errorString.clear();
    return out;
}

bool CommunicationProtocol::bcTestConnection()
{
    // Load communication read options from settings before testing
    loadCommReadOptions();
    
    bool success = testConnection();
    if(!success)
    {
        if(d_errorString.isEmpty() && _device())
            d_errorString = _device()->errorString();
    }

    return success;
}

void CommunicationProtocol::loadCommReadOptions()
{
    SettingsStorage s(d_key, SettingsStorage::Hardware);
    
    // Get current communication type to determine which protocol settings to load
    auto commType = static_cast<CommunicationProtocol::CommType>(
        s.get(BC::Key::HW::commType, static_cast<int>(CommunicationProtocol::Virtual))
    );
    
    // Get the protocol key name for settings lookup
    QString protocolKey;
    switch(commType) {
    case CommunicationProtocol::Rs232:
        protocolKey = BC::Key::Comm::rs232;
        break;
    case CommunicationProtocol::Tcp:
        protocolKey = BC::Key::Comm::tcp;
        break;
    case CommunicationProtocol::Gpib:
        protocolKey = BC::Key::Comm::gpib;
        break;
    case CommunicationProtocol::Custom:
        protocolKey = BC::Key::Comm::custom;
        break;
    case CommunicationProtocol::Virtual:
        protocolKey = BC::Key::Comm::hwVirtual;
        break;
    default:
        // No read options for None or unknown protocols, use defaults
        setReadOptions(1000, "");
        return;
    }
    
    // Load read options using native SettingsStorage group support
    // Use hardware-specific defaults if user hasn't configured protocol settings
    int defaultTimeout = s.get<int>(BC::Key::Comm::timeout, 1000);
    QString defaultTermChar = s.get<QString>(BC::Key::Comm::termChar, QString(""));
    
    int timeout = s.getGroupValue<int>(protocolKey, BC::Key::Comm::timeout, defaultTimeout);
    QString termChar = s.getGroupValue<QString>(protocolKey, BC::Key::Comm::termChar, defaultTermChar);
    
    setReadOptions(timeout, termChar);
}

