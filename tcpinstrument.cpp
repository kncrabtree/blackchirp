#include "tcpinstrument.h"
#include <QTime>

TcpInstrument::TcpInstrument(QString key, QString name, QObject *parent) :
    HardwareObject(key,name,parent)
{
#ifdef BC_NOTCP
	d_hardwareDisabled = true;
#endif
}

TcpInstrument::~TcpInstrument()
{
    disconnectSocket();
}

void TcpInstrument::initialize()
{
	if(d_virtualHardware)
		return;

	d_socket = new QTcpSocket(this);
	connect(d_socket,SIGNAL(error(QAbstractSocket::SocketError)),this,SLOT(socketError(QAbstractSocket::SocketError)));

	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
	QString ip = s.value(key().append(QString("/ip")),QString("")).toString();
	int port = s.value(key().append(QString("/port")),5000).toInt();

	s.setValue(key().append(QString("/prettyName")),name());

	setSocketConnectionInfo(ip,port);
}

bool TcpInstrument::testConnection()
{
	if(d_virtualHardware)
		return true;

	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
	QString ip = s.value(key().append(QString("/ip")),QString("")).toString();
	int port = s.value(key().append(QString("/port")),5000).toInt();

	if(ip == d_ip && port == d_port && d_socket->state() == QTcpSocket::ConnectedState)
		return true;

	if(d_socket->state() != QTcpSocket::UnconnectedState)
		disconnectSocket();

    setSocketConnectionInfo(ip,port);

	return connectSocket();

}

void TcpInstrument::socketError(QAbstractSocket::SocketError)
{
	//consider handling errors here at the socket level
}

bool TcpInstrument::writeCmd(QString cmd)
{
	if(d_virtualHardware)
		return true;

    if(d_socket->state() != QTcpSocket::ConnectedState)
    {
        if(!connectSocket())
        {
            emit hardwareFailure(this);
            emit logMessage(QString("Could not write command to %1. Socket is not connected. (Command = %2)").arg(d_prettyName).arg(cmd));
            return false;
        }
    }

    d_socket->write(cmd.toLatin1());

    if(!d_socket->flush())
    {
        emit hardwareFailure(this);
        emit logMessage(QString("Could not write command to %1. (Command = %2)").arg(d_prettyName).arg(cmd),LogHandler::Error);
        return false;
    }
    return true;
}

QByteArray TcpInstrument::queryCmd(QString cmd)
{
	if(d_virtualHardware)
		return QByteArray();

    if(d_socket->state() != QTcpSocket::ConnectedState)
    {
        if(!connectSocket())
        {
            emit hardwareFailure(this);
            emit logMessage(QString("Could not write query to %1. Socket is not connected. (Query = %2)").arg(d_prettyName).arg(cmd));
            return QByteArray();
        }
    }

    if(d_socket->bytesAvailable())
        d_socket->readAll();

    d_socket->write(cmd.toLatin1());

    if(!d_socket->flush())
    {
        emit hardwareFailure(this);
        emit logMessage(QString("Could not write query to %1. (query = %2)").arg(d_prettyName).arg(cmd),LogHandler::Error);
        return QByteArray();
    }

	//write to socket here, return response
    if(!d_useTermChar || d_readTerminator.isEmpty())
    {
        if(!d_socket->waitForReadyRead(d_timeOut))
        {
            emit hardwareFailure(this);
            emit logMessage(QString("%1 did not respond to query. (query = %2)").arg(d_prettyName).arg(cmd),LogHandler::Error);
            return QByteArray();
        }

        return d_socket->readAll();
    }
    else
    {
        QByteArray out;
        bool done = false;
        while(!done)
        {
            if(!d_socket->waitForReadyRead(d_timeOut))
                break;

            out.append(d_socket->readAll());
            if(out.endsWith(d_readTerminator))
                return out;
        }

        emit hardwareFailure(this);
        emit logMessage(QString("%1 timed out while waiting for termination character. (query = %2, partial response = %3)").arg(d_prettyName).arg(cmd).arg(QString(out)),LogHandler::Error);
        emit logMessage(QString("Hex response: %1").arg(QString(out.toHex())));
        return out;
    }
    return QByteArray();
}

bool TcpInstrument::connectSocket()
{
	if(d_virtualHardware)
		return true;

    d_socket->connectToHost(d_ip,d_port);
    if(!d_socket->waitForConnected(1000))
    {
        emit logMessage(QString("Could not connect to %1 at %2:%3. %4").arg(d_prettyName).arg(d_ip).arg(d_port).arg(d_socket->errorString()),LogHandler::Error);
        return false;
    }
    d_socket->setSocketOption(QAbstractSocket::KeepAliveOption,1);
    d_socket->setSocketOption(QAbstractSocket::LowDelayOption,1);
    return true;
}

void TcpInstrument::disconnectSocket()
{
	if(d_virtualHardware)
		return;

    d_socket->disconnectFromHost();
}

void TcpInstrument::setSocketConnectionInfo(QString ip, int port)
{
	if(d_virtualHardware)
		return;

	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
	s.setValue(key().append(QString("/ip")),ip);
	s.setValue(key().append(QString("/port")),port);
	s.sync();

	d_ip = ip;
	d_port = port;
}
