#ifndef M4I2211X8_H
#define M4I2211X8_H

#include <hardware/core/lifdigitizer/lifscope.h>
#include <hardware/library/spectrumlibrary.h>
#include <hardware/library/spectrumconstants.h>

class QTimer;

class M4i2211x8 : public LifScope
{
    Q_OBJECT
public:
    M4i2211x8(const QString& label, QObject *parent = nullptr);
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

    void* p_handle;

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
