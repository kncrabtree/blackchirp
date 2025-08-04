#ifndef DSOV204A_H
#define DSOV204A_H

#include "ftmwscope.h"

class QTcpSocket;


class DSOv204A : public FtmwScope
{
    Q_OBJECT
public:
    explicit DSOv204A(const QString& label, QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    bool prepareForExperiment(Experiment &exp) override;
    void beginAcquisition() override;
    void endAcquisition() override;

protected:
    void initialize() override;
    bool testConnection() override;

    // FtmwScope interface
public slots:
    void readWaveform() override;

private:
    QTcpSocket *p_socket;
    QTimer *p_queryTimer;
    void retrieveData();
    bool scopeCommand(QString cmd);

    bool d_acquiring{false};
    bool d_processing{false};
};

#endif // DSOV204A_H
