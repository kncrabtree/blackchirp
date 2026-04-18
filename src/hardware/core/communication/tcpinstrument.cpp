#include "tcpinstrument.h"
#include <data/settings/hardwarekeys.h>
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
    
    // Load TCP settings from group with backward compatibility fallback
    d_ip = s.getGroupValue<QString>(BC::Key::Comm::tcp, ip, s.get<QString>(ip, ""));
    d_port = s.getGroupValue<int>(BC::Key::Comm::tcp, port, s.get<int>(port, 5000));

	return connectSocket();

}

bool TcpInstrument::writeCmd(const QString &cmd)
{

    if(p_device->state() != QTcpSocket::ConnectedState)
    {
        if(!connectSocket())
        {
            emit hardwareFailure();
            bcError("Could not write command. Socket is not connected."_L1);
            bcDebug(u"%1 writeCmd: Could not write command. Socket is not connected. Command = %2"_s.arg(d_key, cmd));
            return false;
        }
    }

    return CommunicationProtocol::writeCmd(cmd);
}

bool TcpInstrument::writeBinary(const QByteArray &dat)
{
    if(p_device->state() != QTcpSocket::ConnectedState)
    {
        if(!connectSocket())
        {
            emit hardwareFailure();
            bcError("Could not write binary data. Socket is not connected."_L1);
            bcDebug(u"%1 writeBinary: Could not write binary data. Socket is not connected. Data hex (first 25 bytes) = %2"_s.arg(d_key, QString(dat.toHex()).mid(0,50)));
            return false;
        }
    }

    return CommunicationProtocol::writeBinary(dat);

}

QByteArray TcpInstrument::queryCmd(const QString &cmd, bool suppressError)
{

    if(p_device->state() != QTcpSocket::ConnectedState)
    {
        if(!connectSocket())
        {
            emit hardwareFailure();
            bcError("Could not write query. Socket is not connected."_L1);
            bcDebug(u"%1 queryCmd: Could not write query. Socket is not connected. Query = %2"_s.arg(d_key, cmd));
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
