#ifndef LIFLASER_H
#define LIFLASER_H

#include <hardware/core/hardwareobject.h>
#include <data/lif/lifconversion.h>

namespace BC::Key::LifLaser {
inline constexpr QLatin1StringView units{"units"};
inline constexpr QLatin1StringView decimals{"decimals"};
inline constexpr QLatin1StringView minPos{"minPos"};
inline constexpr QLatin1StringView maxPos{"maxPos"};
inline constexpr QLatin1StringView hasFl{"hasFlashlampControl"};
}

/*!
 * \brief Base class for a tunable LIF excitation laser.
 *
 * The driver hooks setPos()/readPos() operate on the grating fundamental
 * (vacuum wavenumber, cm⁻¹). The public setPosition()/readPosition() slots
 * operate on the output-beam wavenumber (also cm⁻¹, but after any frequency
 * conversion) and bridge between the two via the held LifConversion, which
 * defaults to the identity (output == fundamental) until setConversion() is
 * called.
 */
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
    //! Read the output-beam wavenumber (cm⁻¹), or -1 on error.
    double readPosition();
    //! Move to the given output-beam wavenumber (cm⁻¹); returns readPosition(), or -1 on error.
    double setPosition(double pos);
    bool readFlashLamp();
    bool setFlashLamp(bool en);
    //! Set the active fundamental->output conversion (identity by default).
    void setConversion(const LifConversion &c);

private:
    //! Driver hook: read the grating fundamental (cm⁻¹), or a negative sentinel on error.
    virtual double readPos() =0;
    //! Driver hook: move to the given grating fundamental (cm⁻¹).
    virtual void setPos(double pos) =0;
    virtual bool readFl() =0;

    //This function should return whether setting was successful, not whether it's enabled
    virtual bool setFl(bool en) =0;

    bool d_autoDisable{false};
    //! Fundamental->output conversion applied by readPosition()/setPosition().
    //! Unsynchronized: correct only because every LifLaser slot runs on this
    //! object's own (threaded) affinity, and the only cross-thread writer
    //! (HardwareManager::pushLifConversionToLaser()) always reaches
    //! setConversion() via Qt::BlockingQueuedConnection rather than calling
    //! it directly. A future caller that invokes setConversion() over a
    //! Direct connection (or from another thread without going through
    //! invokeMethod) would race this member against
    //! readPosition()/setPosition() with no lock to catch it.
    LifConversion d_conversion;

protected:
    void hwReadSettings() override final;
    /*!
     * \brief Driver hook called after LifLaser base settings are refreshed. Default is a no-op.
     */
    virtual void lifLaserReadSettings() {}

    /*!
     * \brief Current display unit for the position setting (BC::Key::LifLaser::units).
     */
    BC::LifConv::LaserUnit displayUnit() const;

    // HardwareObject interface
public slots:
    bool hwPrepareForExperiment(Experiment &exp) override final;
    void beginAcquisition() override final;
    void endAcquisition() override final;
};

#endif // LIFLASER_H
