#include <hardware/core/communication/communicationprotocol.h>

CommunicationProtocol::CommunicationProtocol(QString key, QObject *parent) :
    QObject(parent),d_key(key), d_useTermChar(false), d_timeOut(1000)
{

}

CommunicationProtocol::~CommunicationProtocol()
{

}

bool CommunicationProtocol::writeCmd(QString cmd)
{
    if(p_device == nullptr)
        return true;

    if(!p_device->isOpen())
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not write command. Serial port is not open. (Command = %1)").arg(cmd),LogHandler::Error);
        return false;
    }

    qint64 ret = p_device->write(cmd.toLatin1());

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
    if(p_device == nullptr)
        return true;

    if(!p_device->isOpen())
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not write binary data. Serial port is not open. (Data hex = %1)").arg(QString(dat.toHex())),LogHandler::Error);
        return false;
    }

    qint64 ret = p_device->write(dat);

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
    if(p_device == nullptr)
        return QByteArray();

    if(p_device->bytesAvailable())
        p_device->readAll();

    qint64 ret = p_device->write(cmd.toLatin1());

    if(ret == -1)
    {
        if(!suppressError)
        {
            emit hardwareFailure();
            emit logMessage(QString("Could not write query. (query = %1)").arg(cmd),LogHandler::Error);
        }
        return QByteArray();
    }

    while (p_device->bytesToWrite() > 0) {
        if(!p_device->waitForBytesWritten(30000))
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
        if(!p_device->waitForReadyRead(d_timeOut))
        {
            if(!suppressError)
            {
                emit hardwareFailure();
                emit logMessage(QString("Did not respond to query. (query = %1)").arg(cmd),LogHandler::Error);
            }
            return QByteArray();
        }

        return p_device->readAll();
    }
    else
    {
        QByteArray out;
        bool done = false;
        while(!done)
        {
            if(!p_device->waitForReadyRead(d_timeOut))
                break;

            QByteArray t = p_device->readAll();
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
        if(d_errorString.isEmpty() && p_device)
            d_errorString = p_device->errorString();
    }

    return success;
}

