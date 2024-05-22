#ifndef TCPINSTRUMENT_H
#define TCPINSTRUMENT_H

#include <hardware/core/communication/communicationprotocol.h>

#include <QTcpSocket>

namespace BC::Key::TCP {
static const QString ip{"ip"};
static const QString port{"port"};
}

class TcpInstrument : public CommunicationProtocol
{
	Q_OBJECT
public:
    explicit TcpInstrument(QString key, QObject *parent = nullptr);
    ~TcpInstrument();

    bool writeCmd(QString cmd) override;
    bool writeBinary(QByteArray dat) override;
    QByteArray queryCmd(QString cmd, bool suppressError = false) override;

public slots:
    virtual void initialize() override;
    virtual bool testConnection() override;

private:
    QTcpSocket *p_device;
    QString d_ip;
    int d_port;

    bool connectSocket();
    void disconnectSocket();
	
    
    // CommunicationProtocol interface
public:
    QIODevice *_device() override;
};

#endif // TCPINSTRUMENT_H
