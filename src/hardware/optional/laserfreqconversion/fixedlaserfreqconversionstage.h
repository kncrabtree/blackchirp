#ifndef FIXEDLASERFREQCONVERSIONSTAGE_H
#define FIXEDLASERFREQCONVERSIONSTAGE_H

#include <hardware/optional/laserfreqconversion/laserfreqconversionstage.h>

/*!
 * \brief Logical LaserFreqConversionStage for a conversion-topology node not
 *        under Blackchirp's control.
 *
 * Follows the FixedClock motif (hardware/core/clock/fixedclock.h): a real,
 * user-selectable device on CommunicationProtocol::Virtual rather than a
 * placeholder. Its setPosition() is a no-op that always succeeds, so an
 * uncontrolled crystal or compensator still participates in the conversion
 * topology (and therefore the axis math) with no second authoring location.
 */
class FixedLaserFreqConversionStage : public LaserFreqConversionStage
{
    Q_OBJECT
public:
    FixedLaserFreqConversionStage(const QString& label, QObject *parent = nullptr);

protected:
    void initialize() override;
    bool testConnection() override;

    // LaserFreqConversionStage interface
private:
    void setPos(double localCm1) override;
    double readPos() override;

    double d_pos{0.0};
};

#endif // FIXEDLASERFREQCONVERSIONSTAGE_H
