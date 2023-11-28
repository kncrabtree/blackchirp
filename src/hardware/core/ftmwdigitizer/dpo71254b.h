#ifndef DPO71254B_H
#define DPO71254B_H

#include <hardware/core/ftmwdigitizer/ftmwscope.h>

#include <QTimer>
#include <QAbstractSocket>

class QTcpSocket;

namespace BC::Key::FtmwScope {
static const QString dpo71254b{"DPO71254B"};
static const QString dpo71254bName("Ftmw Oscilloscope DPO71254B");
}

class Dpo71254b : public FtmwScope
{
    Q_OBJECT
public:
    explicit Dpo71254b(QObject *parent = nullptr);
    ~Dpo71254b();

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

#endif // DPO71254B_H
