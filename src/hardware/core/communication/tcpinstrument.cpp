#include "tcpinstrument.h"
#include <QTime>

TcpInstrument::TcpInstrument(QString key, QObject *parent) : CommunicationProtocol(key,parent)
{
}

TcpInstrument::~TcpInstrument()
{
    disconnectSocket();
}

void TcpInstrument::initialize()
{
    p_device = new QTcpSocket(this);
}

bool TcpInstrument::testConnection()
{
    using namespace BC::Key::TCP;
    if(p_device->state() == QTcpSocket::ConnectedState)
        disconnectSocket();

    SettingsStorage s(d_key,SettingsStorage::Hardware);
    d_ip = s.get<QString>(ip,"");
    d_port = s.get<int>(port,5000);

	return connectSocket();

}

bool TcpInstrument::writeCmd(QString cmd)
{

    if(p_device->state() != QTcpSocket::ConnectedState)
    {
        if(!connectSocket())
        {
            emit hardwareFailure();
            emit logMessage(QString("Could not write command. Socket is not connected. (Command = %1)").arg(cmd));
            return false;
        }
    }

    return CommunicationProtocol::writeCmd(cmd);
}

bool TcpInstrument::writeBinary(QByteArray dat)
{
    if(p_device->state() != QTcpSocket::ConnectedState)
    {
        if(!connectSocket())
        {
            emit hardwareFailure();
            emit logMessage(QString("Could not write binary data. Socket is not connected. Data hex (first 25 bytes) = %1").arg(QString(dat.toHex()).mid(0,50)),LogHandler::Error);
            return false;
        }
    }

    return CommunicationProtocol::writeBinary(dat);

}

QByteArray TcpInstrument::queryCmd(QString cmd, bool suppressError)
{

    if(p_device->state() != QTcpSocket::ConnectedState)
    {
        if(!connectSocket())
        {
            emit hardwareFailure();
            emit logMessage(QString("Could not write query. Socket is not connected. (Query = %1)").arg(cmd));
            return QByteArray();
        }
    }

    return CommunicationProtocol::queryCmd(cmd, suppressError);

}

bool TcpInstrument::connectSocket()
{
    p_device->connectToHost(d_ip,d_port);
    if(!p_device->waitForConnected(3000))
    {
        setErrorString(QString("Could not connect to %1:%2. %3")
                       .arg(d_ip).arg(d_port).arg(p_device->errorString()));
        return false;
    }
    p_device->setSocketOption(QAbstractSocket::KeepAliveOption,1);

    return true;
}

void TcpInstrument::disconnectSocket()
{
    p_device->disconnectFromHost();
}


QIODevice *TcpInstrument::_device()
{
    return p_device;
}
