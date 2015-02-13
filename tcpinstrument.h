#ifndef TCPINSTRUMENT_H
#define TCPINSTRUMENT_H

#include "hardwareobject.h"
#include <QTcpSocket>

class TcpInstrument : public HardwareObject
{
	Q_OBJECT
public:
	explicit TcpInstrument(QString key, QString name, QObject *parent = 0);
    ~TcpInstrument();
	
signals:
	
public slots:
	virtual void initialize();
	virtual bool testConnection();

	void socketError(QAbstractSocket::SocketError);

protected:
	QTcpSocket *d_socket;
	QString d_ip;
	int d_port;

	bool writeCmd(QString cmd);
    QByteArray queryCmd(QString cmd);
    bool connectSocket();
    void disconnectSocket();


private:
	void setSocketConnectionInfo(QString ip, int port);
	
};

#endif // TCPINSTRUMENT_H
