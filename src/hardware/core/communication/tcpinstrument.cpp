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
    if(dynamic_cast<QTcpSocket*>(p_device)->state() == QTcpSocket::ConnectedState)
        disconnectSocket();

    SettingsStorage s(d_key,SettingsStorage::Hardware);
    d_ip = s.get<QString>(ip,"");
    d_port = s.get<int>(port,5000);

	return connectSocket();

}

bool TcpInstrument::writeCmd(QString cmd)
{

    if(dynamic_cast<QTcpSocket*>(p_device)->state() != QTcpSocket::ConnectedState)
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
    if(dynamic_cast<QTcpSocket*>(p_device)->state() != QTcpSocket::ConnectedState)
    {
        if(!connectSocket())
        {
            emit hardwareFailure();
            emit logMessage(QString("Could not write binary data. Socket is not connected. Data hex (first 25 bytes) = %1").arg(QString(dat.toHex()).mid(0,50)),BlackChirp::LogError);
            return false;
        }
    }

    return CommunicationProtocol::writeBinary(dat);

}

QByteArray TcpInstrument::queryCmd(QString cmd, bool suppressError)
{

    if(dynamic_cast<QTcpSocket*>(p_device)->state() != QTcpSocket::ConnectedState)
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
    auto p_socket = dynamic_cast<QTcpSocket*>(p_device);
    p_socket->connectToHost(d_ip,d_port);
    if(!p_socket->waitForConnected(1000))
    {
        d_errorString = QString("Could not connect to %1:%2. %3").arg(d_ip).arg(d_port).arg(p_device->errorString());
        return false;
    }
    p_socket->setSocketOption(QAbstractSocket::KeepAliveOption,1);

    return true;
}

void TcpInstrument::disconnectSocket()
{
    dynamic_cast<QTcpSocket*>(p_device)->disconnectFromHost();
}
