#ifndef MSO64B_H
#define MSO64B_H

#include "ftmwscope.h"

#include <QTimer>
#include <QAbstractSocket>


class QTcpSocket;

class MSO64B : public FtmwScope
{
    Q_OBJECT
public:
    explicit MSO64B(const QString& label, QObject *parent = nullptr);
    ~MSO64B();

    // HardwareObject interface
public slots:
    bool prepareForExperiment(Experiment &exp) override;
    void beginAcquisition() override;
    void endAcquisition() override;

    void readWaveform() override;
    void wakeUp();
    void socketError(QAbstractSocket::SocketError e);

protected:
    void initialize() override;
    bool testConnection() override;

private:
    bool d_waitingForReply;
    bool d_foundHeader;
    int d_headerNumBytes;
    int d_waveformBytes;
    QTimer *p_scopeTimeout;

    QByteArray scopeQueryCmd(QString query);
    QTcpSocket *p_socket;
};

#endif // MSO64B_H
