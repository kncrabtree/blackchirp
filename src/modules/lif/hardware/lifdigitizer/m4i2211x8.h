#ifndef M4I2211X8_H
#define M4I2211X8_H

#include <modules/lif/hardware/lifdigitizer/lifscope.h>

#include <spcm/dlltyp.h>
#include <spcm/regs.h>
#include <spcm/spcerr.h>
#include <spcm/spcm_drv.h>

class QTimer;

namespace BC::Key::LifDigi {
static const QString m4i2211x8{"m4i2211x8"};
static const QString m4i2211x8Name("Spectrum Instrumentation M4i.2211-x8 Digitizer");
}

class M4i2211x8 : public LifScope
{
    Q_OBJECT
public:
    M4i2211x8(QObject *parent = nullptr);
    ~M4i2211x8() override;

    // HardwareObject interface
protected:
    void initialize() override;
    bool testConnection() override;


    // LifScope interface
public slots:
    void readWaveform() override;

private:
    QTimer *p_timer;

    drv_handle p_handle;

    int d_bufferSize;
    bool errorCheck();

    void startCard();
    void stopCard();

    // LifScope interface
public slots:
    bool configure(const LifDigitizerConfig &c);

    // HardwareObject interface
public slots:
    void beginAcquisition();
    void endAcquisition();
};

#endif // M4I2211X8_H
