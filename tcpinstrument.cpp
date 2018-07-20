#include "tcpinstrument.h"
#include <QTime>

TcpInstrument::TcpInstrument(QString key, QString subKey, QObject *parent) : CommunicationProtocol(CommunicationProtocol::Tcp,key,subKey,parent)
{
}

TcpInstrument::~TcpInstrument()
{
    disconnectSocket();
}

void TcpInstrument::initialize()
{
    p_device = new QTcpSocket(this);

	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
	QString ip = s.value(key().append(QString("/ip")),QString("")).toString();
	int port = s.value(key().append(QString("/port")),5000).toInt();

	setSocketConnectionInfo(ip,port);
}

bool TcpInstrument::testConnection()
{

	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
	QString ip = s.value(key().append(QString("/ip")),QString("")).toString();
	int port = s.value(key().append(QString("/port")),5000).toInt();

    if(ip == d_ip && port == d_port && p_device->isOpen())
		return true;

    disconnectSocket();

    setSocketConnectionInfo(ip,port);

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

QByteArray TcpInstrument::queryCmd(QString cmd)
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

    return CommunicationProtocol::queryCmd(cmd);

}

bool TcpInstrument::connectSocket()
{
    auto p_socket = dynamic_cast<QTcpSocket*>(p_device);
    p_socket->connectToHost(d_ip,d_port);
    if(!p_socket->waitForConnected(1000))
    {
        emit logMessage(QString("Could not connect to %1:%2. %3").arg(d_ip).arg(d_port).arg(p_device->errorString()),BlackChirp::LogError);
        return false;
    }
    p_socket->setSocketOption(QAbstractSocket::KeepAliveOption,1);
//    p_socket->setSocketOption(QAbstractSocket::LowDelayOption,1);
    return true;
}

void TcpInstrument::disconnectSocket()
{
    dynamic_cast<QTcpSocket*>(p_device)->disconnectFromHost();
}

void TcpInstrument::setSocketConnectionInfo(QString ip, int port)
{
	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
	s.setValue(key().append(QString("/ip")),ip);
	s.setValue(key().append(QString("/port")),port);
	s.sync();

	d_ip = ip;
	d_port = port;
}
