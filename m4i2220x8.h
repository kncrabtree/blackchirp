#ifndef M412220X8_H
#define M412220X8_H

#include "ftmwscope.h"

#include "spcm/dlltyp.h"
#include "spcm/regs.h"
#include "spcm/spcerr.h"
#include "spcm/spcm_drv.h"

#include <QTimer>


class M4i2220x8 : public FtmwScope
{
    Q_OBJECT
public:
    explicit M4i2220x8(QObject *parent = nullptr);
    ~M4i2220x8();

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
    drv_handle p_handle;

    qint64 d_waveformBytes;
    QByteArray d_m4iBuffer;

    QTimer *p_timer;

};

#endif // M412220X8_H
