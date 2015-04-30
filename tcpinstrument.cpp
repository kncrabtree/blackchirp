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
	d_socket = new QTcpSocket(this);
	connect(d_socket,SIGNAL(error(QAbstractSocket::SocketError)),this,SLOT(socketError(QAbstractSocket::SocketError)));

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

    if(d_socket->state() != QTcpSocket::ConnectedState)
    {
        if(!connectSocket())
        {
            emit hardwareFailure();
            emit logMessage(QString("Could not write command. Socket is not connected. (Command = %1)").arg(cmd));
            return false;
        }
    }

    d_socket->write(cmd.toLatin1());

    if(!d_socket->flush())
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not write command. (Command = %1)").arg(cmd),BlackChirp::LogError);
        return false;
    }
    return true;
}

QByteArray TcpInstrument::queryCmd(QString cmd)
{

    if(d_socket->state() != QTcpSocket::ConnectedState)
    {
        if(!connectSocket())
        {
            emit hardwareFailure();
            emit logMessage(QString("Could not write query. Socket is not connected. (Query = %1)").arg(cmd));
            return QByteArray();
        }
    }

    if(d_socket->bytesAvailable())
        d_socket->readAll();

    d_socket->write(cmd.toLatin1());

    if(!d_socket->flush())
    {
        emit hardwareFailure();
        emit logMessage(QString("Could not write query. (query = %1)").arg(cmd),BlackChirp::LogError);
        return QByteArray();
    }

	//write to socket here, return response
    if(!d_useTermChar || d_readTerminator.isEmpty())
    {
        if(!d_socket->waitForReadyRead(d_timeOut))
        {
            emit hardwareFailure();
            emit logMessage(QString("Did not respond to query. (query = %1)").arg(cmd),BlackChirp::LogError);
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

        emit hardwareFailure();
        emit logMessage(QString("Timed out while waiting for termination character. (query = %1, partial response = %2)").arg(cmd).arg(QString(out)),BlackChirp::LogError);
        emit logMessage(QString("Hex response: %1").arg(QString(out.toHex())));
        return out;
    }
    return QByteArray();
}

bool TcpInstrument::connectSocket()
{
    d_socket->connectToHost(d_ip,d_port);
    if(!d_socket->waitForConnected(1000))
    {
        emit logMessage(QString("Could not connect to %1:%2. %3").arg(d_ip).arg(d_port).arg(d_socket->errorString()),BlackChirp::LogError);
        return false;
    }
    d_socket->setSocketOption(QAbstractSocket::KeepAliveOption,1);
    d_socket->setSocketOption(QAbstractSocket::LowDelayOption,1);
    return true;
}

void TcpInstrument::disconnectSocket()
{
    d_socket->disconnectFromHost();
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
