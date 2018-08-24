#ifndef DSOX92004A_H
#define DSOX92004A_H

#include "ftmwscope.h"

class QTcpSocket;

class DSOx92004A : public FtmwScope
{
    Q_OBJECT
public:
    explicit DSOx92004A(QObject *parent = nullptr);

    // HardwareObject interface
public slots:
    bool testConnection();
    void initialize();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();
    void readTimeData();

    // FtmwScope interface
public slots:
    void readWaveform();

private:
    QTcpSocket *p_socket;
    QTimer *p_queryTimer;
    void retrieveData();

    bool d_acquiring;
};

#endif // DSOX92004A_H
