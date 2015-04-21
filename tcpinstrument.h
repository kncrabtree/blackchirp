#ifndef TCPINSTRUMENT_H
#define TCPINSTRUMENT_H

#include "communicationprotocol.h"

#include <QTcpSocket>

class TcpInstrument : public CommunicationProtocol
{
	Q_OBJECT
public:
    explicit TcpInstrument(QString key, QString subKey, QObject *parent = nullptr);
    ~TcpInstrument();

    bool writeCmd(QString cmd);
    QByteArray queryCmd(QString cmd);
    QTcpSocket *d_socket;

public slots:
	virtual void initialize();
	virtual bool testConnection();
	void socketError(QAbstractSocket::SocketError);


private:
    QString d_ip;
    int d_port;

    bool connectSocket();
    void disconnectSocket();
	void setSocketConnectionInfo(QString ip, int port);
	
};

#endif // TCPINSTRUMENT_H
