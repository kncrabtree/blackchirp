#ifndef LIFLASER_H
#define LIFLASER_H

#include <hardware/core/hardwareobject.h>

namespace BC::Key::LifLaser {
inline constexpr QLatin1StringView units{"units"};
inline constexpr QLatin1StringView decimals{"decimals"};
inline constexpr QLatin1StringView minPos{"minPos"};
inline constexpr QLatin1StringView maxPos{"maxPos"};
inline constexpr QLatin1StringView hasFl{"hasFlashlampControl"};
}

class LifLaser : public HardwareObject
{
    Q_OBJECT
public:
    LifLaser(const QString& impl, const QString& label, QObject *parent = nullptr);
    ~LifLaser() override;

signals:
    void laserPosUpdate(double);
    void laserFlashlampUpdate(bool);

public slots:
    double readPosition();
    double setPosition(double pos);
    bool readFlashLamp();
    bool setFlashLamp(bool en);

private:
    virtual double readPos() =0;
    virtual void setPos(double pos) =0;
    virtual bool readFl() =0;

    //This function should return whether setting was successful, not whether it's enabled
    virtual bool setFl(bool en) =0;

    bool d_autoDisable{false};

protected:
    void hwReadSettings() override final;
    /*!
     * \brief Driver hook called after LifLaser base settings are refreshed. Default is a no-op.
     */
    virtual void lifLaserReadSettings() {}

    // HardwareObject interface
public slots:
    bool hwPrepareForExperiment(Experiment &exp) override final;
    void beginAcquisition() override final;
    void endAcquisition() override final;
};

#endif // LIFLASER_H
