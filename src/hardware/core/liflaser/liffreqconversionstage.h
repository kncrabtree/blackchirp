#ifndef LIFFREQCONVERSIONSTAGE_H
#define LIFFREQCONVERSIONSTAGE_H

#include <hardware/core/hardwareobject.h>
#include <data/storage/settingsstorage.h>
#include <data/lif/lifconversion.h>

namespace BC::Key::LifConvStage {
inline constexpr QLatin1StringView op{"conversionOp"};          ///< BC::LifConv::Op key name.
inline constexpr QLatin1StringView harmonic{"harmonicOrder"};   ///< int N, NHG only.
inline constexpr QLatin1StringView isFinal{"finalBeam"};        ///< bool.
inline constexpr QLatin1StringView verify{"verifyMove"};        ///< bool, default true.
inline constexpr QLatin1StringView tolerance{"verifyToleranceCm1"}; ///< double cm⁻¹, move-verification window.
inline constexpr QLatin1StringView inputs{"conversionInputs"};  ///< array.
// inputs[] entry subkeys:
inline constexpr QLatin1StringView refType{"refType"};          ///< RefType key name.
inline constexpr QLatin1StringView refKey{"refStageKey"};       ///< stage hwKey when Stage.
inline constexpr QLatin1StringView refFixedCm1{"fixedCm1"};     ///< when Fixed.
}

/*!
 * \brief Base class for a LIF frequency-conversion stage (FCU): a crystal or
 *        compensator node in the optical conversion topology between the
 *        tunable LifLaser fundamental and the FINAL output beam.
 *
 * A direct child of HardwareObject (sibling of LifLaser), so it earns its
 * own hwType. The base owns only the generic contract shared by every
 * conversion node: a registered node descriptor (conversionNode(), read into
 * a BC::LifConv::Node — op, harmonic order, ordered input refs, FINAL
 * marker), a per-device verify flag, and the setPosition()/readPosition()
 * dispatch slots that move to and confirm a local input-beam wavenumber
 * (cm⁻¹). Structured calibration settings (crystal/compensator angle-vs-
 * wavelength polynomials, mount addresses) differ by driver and are not
 * declared here.
 *
 * A stage emits no output-position update to the display; only
 * LifLaser::laserPosUpdate drives the axis.
 */
class LifFreqConversionStage : public HardwareObject
{
    Q_OBJECT
public:
    LifFreqConversionStage(const QString& impl, const QString& label, QObject *parent = nullptr);
    ~LifFreqConversionStage() override;

    //! Read this stage's registered node descriptor into a BC::LifConv::Node (stageKey == d_key).
    BC::LifConv::Node conversionNode() const;

    /*!
     * \brief Read a BC::Key::LifConvStage node descriptor out of an
     *        arbitrary SettingsStorage snapshot rather than a live device.
     *
     * Shared by conversionNode() (\a s == *this, live settings) and by
     * GUI-thread callers that assemble a conversion from settings
     * snapshots (SettingsStorage constructed directly on a hardware key)
     * without touching the threaded device.
     *
     * \param s Settings snapshot to read from.
     * \param stageKey Value written into the returned Node's stageKey.
     */
    static BC::LifConv::Node nodeFromSettings(const SettingsStorage &s, const QString &stageKey);

public slots:
    /*!
     * \brief Dispatch target: move to this stage's PRIMARY input-beam
     *        wavenumber (cm⁻¹).
     *
     * \a localCm1 is already computed by the caller from the assembled
     * conversion topology (LifConversion::stageInput); the stage itself
     * needs no topology. Maps the requested wavenumber to a phase-match
     * motor position via the driver's calibration, moves, then verifies via
     * readPos(). When the verify flag (BC::Key::LifConvStage::verify) is
     * off, a verification mismatch is logged as a warning and the call
     * still returns true (best-effort); when it is on, a mismatch or a
     * negative readPos() emits hardwareFailure() and returns false.
     *
     * \return Whether the move (and, if enabled, its verification) succeeded.
     */
    bool setPosition(double localCm1);

    //! Verify hook: this stage's achieved local input-beam wavenumber (cm⁻¹), or <0 on error.
    double readPosition();

private:
    //! Driver hook: move to the given local input-beam wavenumber (cm⁻¹).
    virtual void setPos(double localCm1) = 0;
    //! Driver hook: read the achieved local input-beam wavenumber (cm⁻¹), or a negative sentinel on error.
    virtual double readPos() = 0;
};

/*!
 * \brief Assemble the active LIF frequency-conversion topology from
 *        settings snapshots, without touching any threaded device.
 *
 * Reads RuntimeHardwareConfig::constInstance().getActiveKeys<LifFreqConversionStage>(),
 * builds a BC::LifConv::Node for each from a settings snapshot (a
 * SettingsStorage constructed directly on the stage's hardware key) via
 * LifFreqConversionStage::nodeFromSettings(), and assembles them with
 * LifConversion::assemble().
 *
 * Intended for GUI-thread callers (config page, live laser widget) that
 * need the active topology to compute display ranges but must not block
 * on a threaded device. Config-time topology may be mid-edit and
 * transiently invalid; the authoritative validator is HardwareManager's
 * prep-time assemble(), which aborts the experiment on failure. Here, a
 * failed assemble() instead falls back to the default identity
 * LifConversion so the GUI stays responsive and simply shows the
 * unconverted fundamental range until the topology is fixed.
 *
 * \return The assembled conversion, or the identity conversion when
 * assembly fails or no stages are active.
 */
LifConversion assembleActiveLifConversion();

#endif // LIFFREQCONVERSIONSTAGE_H
