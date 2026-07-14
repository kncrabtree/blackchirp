#ifndef SIRAHFCU_H
#define SIRAHFCU_H

#include <hardware/core/liflaser/liffreqconversionstage.h>
#include <hardware/core/liflaser/autotrackerprotocol.h>
#include <data/lif/fcucalibration.h>

namespace BC::Key::SirahFcu {
inline constexpr QLatin1StringView motorNumber{"motorNumber"};
inline constexpr QLatin1StringView stages{"stages"};
inline constexpr QLatin1StringView sStart{"stageStartFreqHz"};
inline constexpr QLatin1StringView sHigh{"stageHighFreqHz"};
inline constexpr QLatin1StringView sRamp{"stageRampLength"};
inline constexpr QLatin1StringView sMax{"stageMaxPos"};
inline constexpr QLatin1StringView sbls{"stageBacklashSteps"};
inline constexpr QLatin1StringView sLeverLength{"stageLeverLengthMm"};
inline constexpr QLatin1StringView sLinearOffset{"stageLinearOffsetMm"};
inline constexpr QLatin1StringView sAngleOffset{"stageAngleOffsetDeg"};
inline constexpr QLatin1StringView sCutAngle{"stageCutAngleDeg"};
inline constexpr QLatin1StringView sTemperature{"stageTemperature"};
inline constexpr QLatin1StringView sPitch{"stageScrewPitchmmPerRev"};
inline constexpr QLatin1StringView sMotorResolution{"stageMotorResolutionStepsPerRev"};
inline constexpr QLatin1StringView calScheme{"calibrationScheme"};
inline constexpr QLatin1StringView calCrystal{"crystalType"};
inline constexpr QLatin1StringView calInvert{"invertPhaseMatch"};
inline constexpr QLatin1StringView polyCoeffs{"polyCoeffs"};
inline constexpr QLatin1StringView pcOrder{"order"};
inline constexpr QLatin1StringView pcForward{"forward"};
inline constexpr QLatin1StringView pcInverse{"inverse"};
inline constexpr QLatin1StringView splinePoints{"splinePoints"};
inline constexpr QLatin1StringView spWavelength{"wavelengthNm"};
inline constexpr QLatin1StringView spPosition{"positionSteps"};
}

/*!
 * \brief Sirah Frequency Conversion Unit (FCU) driver: a doubling-crystal
 *        stage in the LIF conversion topology.
 *
 * A Sirah Autotracker unit on its own RS232 port, entirely separate from
 * the Cobra grating controller both physically and at the protocol level:
 * it speaks the Autotracker binary command/response protocol
 * (BC::Autotracker::buildCommand()/parseResponse(), see
 * autotrackerprotocol.h) rather than the Cobra's (BC::Sirah, see
 * sirahprotocol.h). The doubling crystal's angle <-> wavelength law is not
 * diffraction, unlike the Cobra's grating: it is evaluated by a
 * FcuCalibration assembled in hwReadSettings() from the registered
 * calibration scheme (see data/lif/fcucalibration.h).
 *
 * The Autotracker addresses up to three motors per unit; \c motorNumber
 * selects which one drives this doubling crystal (default 1).
 *
 * The registered harmonic-order default is overridden for the common
 * lone-doubler case: N defaults to 2 and is Required (set once at profile
 * creation). This is an NHG device by identity — conversionOp() is pinned
 * to Op::NHG — though the base op setting (already defaulting to NHG)
 * stays registered so it remains snapshot-visible.
 */
class SirahFcu : public LifFreqConversionStage
{
    Q_OBJECT
public:
    explicit SirahFcu(const QString& label, QObject *parent = nullptr);

    // LifFreqConversionStage interface
    BC::LifConv::Op conversionOp() const override;

    // HardwareObject interface
protected:
    void initialize() override;
    bool testConnection() override;
    void hwReadSettings() override;

    // LifFreqConversionStage interface
private:
    void setPos(double localCm1) override;
    double readPos() override;

    //! Position (steps) last reported by the motor, per prompt()/readPos().
    quint32 d_lastPos{0};
    //! Direction (+1/-1, 0 = unknown) of the most recent backlash-compensation move; see setPos().
    int d_lastMoveDir{0};
    FcuCalibration d_calibration;

    //! Generous upper bound on the time a Wait=0 Goto Position ack can take to arrive (full-travel moves can take seconds).
    static constexpr int moveAckTimeoutMs = 15000;

    quint8 motor() const;
    bool prompt();
    void moveRelative(qint32 steps);
    bool moveAbsolute(quint32 targetPos);

    //! Reads a 12-byte Autotracker response frame, retrying on p_comm until \a totalTimeoutMs has elapsed or a full frame arrives.
    QByteArray readResponse(int totalTimeoutMs);
    //! Drains the Autotracker's startup error queue (Error command, 0x03), logging any stale codes found. \return false on a comm failure.
    bool drainErrorQueue();
    //! Issues an Error query and folds any nonzero codes into an hwError() report for \a context.
    void reportCommError(const QString &context);
};

#endif // SIRAHFCU_H
