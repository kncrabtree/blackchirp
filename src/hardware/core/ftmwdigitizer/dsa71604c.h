#ifndef DSA71604C_H
#define DSA71604C_H

#include <hardware/core/ftmwdigitizer/ftmwscope.h>

#include <QTimer>
#include <QAbstractSocket>

class QTcpSocket;

namespace BC::Key::FtmwScope {
static const QString dsa71604c{"dsa71604c"};
static const QString dsa71064cName("Ftmw Oscilloscope DSA71604C");
}

class Dsa71604c : public FtmwScope
{
    Q_OBJECT
public:
    Dsa71604c(QObject *parent = nullptr);
    ~Dsa71604c();

    // HardwareObject interface
public slots:
    bool prepareForExperiment(Experiment &exp) override;
    void beginAcquisition() override;
    void endAcquisition() override;

    void readWaveform() override;
    void wakeUp();
    void socketError(QAbstractSocket::SocketError e);

protected:
    bool testConnection() override;
    void initialize() override;

private:
    bool d_waitingForReply;
    bool d_foundHeader;
    int d_headerNumBytes;
    int d_waveformBytes;
    QTimer *p_scopeTimeout;

    QByteArray scopeQueryCmd(QString query);
    QTcpSocket *p_socket;
};

#endif // DSA71604C_H
