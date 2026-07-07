#ifndef FIXEDLIFFREQCONVERSIONSTAGE_H
#define FIXEDLIFFREQCONVERSIONSTAGE_H

#include <hardware/core/liflaser/liffreqconversionstage.h>

/*!
 * \brief Logical LifFreqConversionStage for a conversion-topology node not
 *        under Blackchirp's control.
 *
 * Follows the FixedClock motif (hardware/core/clock/fixedclock.h): a real,
 * user-selectable device on CommunicationProtocol::Virtual rather than a
 * placeholder. Its setPosition() is a no-op that always succeeds, so an
 * uncontrolled crystal or compensator still participates in the conversion
 * topology (and therefore the axis math) with no second authoring location.
 */
class FixedLifFreqConversionStage : public LifFreqConversionStage
{
    Q_OBJECT
public:
    FixedLifFreqConversionStage(const QString& label, QObject *parent = nullptr);

protected:
    void initialize() override;
    bool testConnection() override;

    // LifFreqConversionStage interface
private:
    void setPos(double localCm1) override;
    double readPos() override;

    double d_pos{0.0};
};

#endif // FIXEDLIFFREQCONVERSIONSTAGE_H
