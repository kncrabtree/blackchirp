#include <hardware/core/communication/communicationprotocol.h>

CommunicationProtocol::CommunicationProtocol(QString key, QObject *parent) :
    QObject(parent),d_key(key)
{

}

CommunicationProtocol::~CommunicationProtocol()
{

}

bool CommunicationProtocol::writeCmd(QString cmd)
{
    if(_device() == nullptr)
        return true;

    if(!_device()->isOpen())
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not write command. Serial port is not open. (Command = %1)").arg(cmd),LogHandler::Error);
        return false;
    }

    qint64 ret = _device()->write(cmd.toLatin1());

    if(ret == -1)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not write command. (Command = %1)").arg(cmd),LogHandler::Error);
        return false;
    }
    return true;
}

bool CommunicationProtocol::writeBinary(QByteArray dat)
{
    if(_device() == nullptr)
        return true;

    if(!_device()->isOpen())
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not write binary data. Serial port is not open. (Data hex = %1)").arg(QString(dat.toHex())),LogHandler::Error);
        return false;
    }

    qint64 ret = _device()->write(dat);

    if(ret == -1)
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not write binary data. (Data hex = %1)").arg(QString(dat.toHex())),LogHandler::Error);
        return false;
    }
    return true;
}

QByteArray CommunicationProtocol::queryCmd(QString cmd, bool suppressError)
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
            emit logMessage(QString("Could not write query. (query = %1)").arg(cmd),LogHandler::Error);
        }
        return QByteArray();
    }

    while (_device()->bytesToWrite() > 0) {
        if(!_device()->waitForBytesWritten(30000))
        {
            if(!suppressError)
            {
                emit hardwareFailure();
                emit logMessage(QString("Timed out while waiting for query write. Query = %1").arg(cmd),LogHandler::Error);
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
                emit logMessage(QString("Did not respond to query. (query = %1)").arg(cmd),LogHandler::Error);
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
            emit logMessage(QString("Timed out while waiting for termination character. (query = %1, partial response = %2)").arg(cmd).arg(QString(out)),LogHandler::Error);
            emit logMessage(QString("Hex response: %1").arg(QString(out.toHex())));
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
                emit logMessage(QString("Could not read %1 bytes; timeout error").arg(n),LogHandler::Error);
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
    bool success = testConnection();
    if(!success)
    {
        if(d_errorString.isEmpty() && _device())
            d_errorString = _device()->errorString();
    }

    return success;
}

