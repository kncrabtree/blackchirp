#ifndef DSOX92004A_H
#define DSOX92004A_H

#include <hardware/core/ftmwdigitizer/ftmwdigitizer.h>

class QTcpSocket;


class DSOx92004A : public FtmwDigitizer
{
    Q_OBJECT
public:
    explicit DSOx92004A(const QString& label, QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    bool prepareForExperiment(Experiment &exp) override;
    void beginAcquisition() override;
    void endAcquisition() override;

    // FtmwDigitizer interface
    void readWaveform() override;

protected:
    bool testConnection() override;
    void initialize() override;


private:
    QTcpSocket *p_socket;
    QTimer *p_queryTimer;
    void retrieveData();
    bool scopeCommand(const QString &cmd);

    bool d_acquiring;
};

#endif // DSOX92004A_H
