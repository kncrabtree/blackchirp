#ifndef DSA71604C_H
#define DSA71604C_H

#include "ftmwscope.h"
#include "tcpinstrument.h"

class Dsa71604c : public FtmwScope
{
    Q_OBJECT
public:
    Dsa71604c(QObject *parent = nullptr);
    ~Dsa71604c();

    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();

    void readWaveform();
    void queryScope();
    void wakeUp();

private:
    bool d_waitingForReply;
    bool d_foundHeader;
    int d_headerNumBytes;
    int d_waveformBytes;
    QDateTime d_lastTrigger;
    bool d_waitingForWakeUp;
    QTimer d_scopeTimeout;

    QByteArray scopeQueryCmd(QString query);
    QTcpSocket *d_socket;
};

#endif // DSA71604C_H
