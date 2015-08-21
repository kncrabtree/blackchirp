#ifndef MSO72004C_H
#define MSO72004C_H

#include "ftmwscope.h"

#include <QTimer>
#include <QAbstractSocket>

class QTcpSocket;

class MSO72004C : public FtmwScope
{
    Q_OBJECT
public:
    MSO72004C(QObject *parent = nullptr);
    ~MSO72004C();

    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();

    void readWaveform();
    void wakeUp();
    void socketError(QAbstractSocket::SocketError e);

private:
    bool d_waitingForReply;
    bool d_foundHeader;
    int d_headerNumBytes;
    int d_waveformBytes;
    QTimer *p_scopeTimeout;

    QByteArray scopeQueryCmd(QString query);
    QTcpSocket *p_socket;
};

#endif // MSO72004C_H
