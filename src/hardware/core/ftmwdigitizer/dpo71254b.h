#ifndef DPO71254B_H
#define DPO71254B_H

#include <hardware/core/ftmwdigitizer/ftmwscope.h>

#include <QTimer>
#include <QAbstractSocket>

class QTcpSocket;


class Dpo71254b : public FtmwScope
{
    Q_OBJECT
public:
    Dpo71254b(const QString& label, QObject *parent = nullptr);
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
    bool testConnection() override;
    void initialize() override;

private:
    bool d_waitingForReply;
    bool d_foundHeader;
    int d_headerNumBytes;
    int d_waveformBytes;
    QTimer *p_scopeTimeout;

    QByteArray scopeQueryCmd(const QString &query);
    QTcpSocket *p_socket;
};

#endif // DPO71254B_H
