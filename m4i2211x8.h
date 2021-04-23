#ifndef M4I2211X8_H
#define M4I2211X8_H

#include "lifscope.h"

#include "spcm/dlltyp.h"
#include "spcm/regs.h"
#include "spcm/spcerr.h"
#include "spcm/spcm_drv.h"

class QTimer;

class M4i2211x8 : public LifScope
{
    Q_OBJECT
public:
    M4i2211x8(QObject *parent = nullptr);
    ~M4i2211x8() override;

    // HardwareObject interface
protected:
    void readSettings() override;
    void initialize() override;
    bool testConnection() override;


    // LifScope interface
public slots:
    void setLifVScale(double scale) override;
    void setRefVScale(double scale) override;
    void setHorizontalConfig(double sampleRate, int recLen) override;
    void setRefEnabled(bool en) override;
    void queryScope() override;

private:
    QTimer *p_timer;

    drv_handle p_handle;

    char *p_m4iBuffer;
    int d_bufferSize;
    int d_timerInterval;
    bool d_running;
    bool errorCheck();
    void configureMemory();

    void startCard();
    bool stopCard();
};

#endif // M4I2211X8_H
