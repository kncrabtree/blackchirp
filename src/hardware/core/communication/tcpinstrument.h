#ifndef TCPINSTRUMENT_H
#define TCPINSTRUMENT_H

#include <src/hardware/core/communication/communicationprotocol.h>

#include <QTcpSocket>

class TcpInstrument : public CommunicationProtocol
{
	Q_OBJECT
public:
    explicit TcpInstrument(QString key, QString subKey, QObject *parent = nullptr);
    ~TcpInstrument();

    bool writeCmd(QString cmd) override;
    bool writeBinary(QByteArray dat) override;
    QByteArray queryCmd(QString cmd, bool suppressError = false) override;

public slots:
    virtual void initialize() override;
    virtual bool testConnection() override;

private:
    QString d_ip;
    int d_port;

    bool connectSocket();
    void disconnectSocket();
	void setSocketConnectionInfo(QString ip, int port);
	
};

#endif // TCPINSTRUMENT_H
