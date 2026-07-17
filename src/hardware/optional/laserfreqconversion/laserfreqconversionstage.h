#ifndef LASERFREQCONVERSIONSTAGE_H
#define LASERFREQCONVERSIONSTAGE_H

#include <hardware/core/hardwareobject.h>
#include <data/storage/settingsstorage.h>
#include <data/lif/lifconversion.h>
#include <data/loadout/lifconversionsnapshot.h>

namespace BC::Key::LaserConvStage {
inline constexpr QLatin1StringView op{"conversionOp"};          ///< BC::LifConv::Op key name.
inline constexpr QLatin1StringView harmonic{"harmonicOrder"};   ///< int N, NHG only.
inline constexpr QLatin1StringView verify{"verifyMove"};        ///< bool, default true.
inline constexpr QLatin1StringView tolerance{"verifyToleranceCm1"}; ///< double cm⁻¹, move-verification window.
}

/*!
 * \brief Base class for a LIF frequency-conversion stage (FCU): a crystal or
 *        compensator node in the optical conversion topology between the
 *        tunable LifLaser fundamental and the FINAL output beam.
 *
 * A direct child of HardwareObject (sibling of LifLaser), so it earns its
 * own hwType. The base owns the generic contract shared by every
 * conversion node: the registered device-identity settings (conversionOp(),
 * harmonicOrder() — op/harmonic order, snapshot-visible so a GUI/data-layer
 * caller can read them without touching the threaded device), a per-device
 * verify flag, and the setPosition()/readPosition() dispatch slots that move
 * to and confirm a local input-beam wavenumber (cm⁻¹). The DAG wiring
 * (input refs, FINAL marker) is per-experiment state owned by
 * LifConversionSnapshot/LifConfig, not by the stage. Structured calibration
 * settings (crystal/compensator angle-vs-wavelength polynomials, mount
 * addresses) differ by driver and are not declared here.
 *
 * A stage emits no output-position update to the display; only
 * LifLaser::laserPosUpdate drives the axis.
 */
class LaserFreqConversionStage : public HardwareObject
{
    Q_OBJECT
public:
    LaserFreqConversionStage(const QString& impl, const QString& label, QObject *parent = nullptr);
    ~LaserFreqConversionStage() override;

    /*!
     * \brief This stage's conversion operation (NHG/SFG/DFG).
     *
     * Base implementation reads the registered \c op setting. A doubler or
     * mixer driver *is* that operation by device identity — not a free
     * choice — so concrete drivers override this to a constant; the
     * registered setting itself stays declared on the base for every stage
     * (concrete drivers pin its default) so it remains snapshot-visible even
     * where the driver treats it as fixed (see the C-6 assembly join, which
     * reads settings snapshots rather than calling this virtual on a live
     * threaded device).
     */
    virtual BC::LifConv::Op conversionOp() const;

    //! Harmonic order N for an NHG stage (the registered \c harmonic setting; ignored for SFG/DFG).
    int harmonicOrder() const { return get(BC::Key::LaserConvStage::harmonic, 2); }

    /*!
     * \brief Driver hook for a gated harmonic-order change.
     *
     * Base implementation persists \a n to the registered \c harmonic
     * setting and returns \c true. A unit that can retune its harmonic
     * output in firmware overrides this to issue the hardware command (and
     * update the setting on success), so a harmonic change always passes
     * through the device rather than being a raw setting poke.
     *
     * \return Whether the change succeeded.
     */
    virtual bool setHarmonicOrder(int n);

public slots:
    /*!
     * \brief Dispatch target: move to this stage's PRIMARY input-beam
     *        wavenumber (cm⁻¹).
     *
     * \a localCm1 is already computed by the caller from the assembled
     * conversion topology (LifConversion::stageInput); the stage itself
     * needs no topology. Maps the requested wavenumber to a phase-match
     * motor position via the driver's calibration, moves, then verifies via
     * readPos(). A negative readPos() is a hard communication-error
     * sentinel, not a value the stage actually reported, so it always
     * emits hardwareFailure() and returns false regardless of the verify
     * flag. For an in-range readback that simply misses the requested
     * wavenumber, the verify flag (BC::Key::LaserConvStage::verify) decides
     * the outcome: off logs a warning and returns true (best-effort); on
     * emits hardwareFailure() and returns false.
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
 * \brief Join \a snap's wiring with op/n read from each stage's active
 *        hardware settings snapshot into a node list, without touching any
 *        threaded device.
 *
 * For every BC::LifConv::StageWiring entry, builds a SettingsStorage
 * snapshot directly on the stage's hardware key (never a live device) to
 * read conversionOp()/harmonicOrder(). Shared by assembleLifConversion()
 * and by callers that need the joined node list itself (e.g. the conversion
 * table model, when seeding wiring from a preset snapshot).
 *
 * \return The joined node list; empty when \a snap has no wiring.
 */
std::vector<BC::LifConv::Node> lifConversionNodesFromSnapshot(const LifConversionSnapshot &snap);

/*!
 * \brief Assemble a LIF frequency-conversion topology by joining \a snap's
 *        wiring with op/n read from each stage's active hardware settings
 *        snapshot, without touching any threaded device.
 *
 * Builds the node list via lifConversionNodesFromSnapshot() and assembles it
 * with LifConversion::assemble(). Snapshot-only, so it is safe to call from
 * GUI or data-layer threads.
 *
 * \return The assembly result: \c ok and \c conversion on success, or \c ok
 * == false and \c errorString on a malformed topology.
 */
LifConversion::AssemblyResult assembleLifConversion(const LifConversionSnapshot &snap);

/*!
 * \brief Convenience wrapper: assemble the current LIF preset's conversion
 *        wiring for the current loadout.
 *
 * Resolves LoadoutManager::instance().currentLoadoutName() and that
 * loadout's current LIF preset. When no loadout or no LIF preset is
 * selected, returns the identity AssemblyResult (\c ok == true, identity
 * LifConversion) rather than an error, matching the tolerant fallback used
 * by GUI callers (config page, live laser widget) that need the active
 * topology to compute display ranges but must stay responsive when the
 * topology is not yet configured.
 *
 * \return The assembly result for the current preset, or the identity
 * result when no preset is selected.
 */
LifConversion::AssemblyResult assembleCurrentLifConversion();

#endif // LASERFREQCONVERSIONSTAGE_H
