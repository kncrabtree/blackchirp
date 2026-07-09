#ifndef LIFCONVERSION_H
#define LIFCONVERSION_H

#include <map>
#include <utility>
#include <vector>

#include <QString>

#include <data/lif/lifunits.h>

/*!
 * \file lifconversion.h
 * \brief Node-descriptor DAG and the \c LifConversion value type that
 *        assembles and evaluates it. All values are vacuum wavenumber
 *        (cm⁻¹); see lifunits.h for the display-unit boundary.
 *
 * \c BC::LifConv::Op and \c BC::LifConv::RefType — the enums this DAG is
 * built from — are declared in lifunits.h alongside \c LaserUnit rather
 * than here; see the comment on \c BC::LifConv::Op there for why a single
 * \c Q_NAMESPACE-tagged namespace cannot have its \c Q_ENUM_NS entries
 * split across two moc-processed headers.
 */
namespace BC::LifConv {

/*!
 * \brief Implementation detail of \c LifConversion — not part of the
 *        frozen public interface. Coefficients of an affine expression
 *        \c value = a*fundamentalCm1 + b, used internally to represent
 *        every beam in an assembled conversion graph.
 *
 * A plain namespace-scope type (rather than a private nested class of \c
 * LifConversion) so the free assembly helpers in lifconversion.cpp can
 * share it without needing friendship.
 */
namespace detail {
struct AffineCoeffs {
    double a{1.0};
    double b{0.0};
};
}

/*!
 * \brief One ordered input to a \c Node. For a two-input \c Op the beam
 *        carrying the tunable dependence may occupy either slot; \c assemble()
 *        identifies it. \c inputs[0] is used as the primary only to
 *        disambiguate a stage whose inputs are all tunable or all Fixed.
 */
struct InputRef {
    RefType type{RefType::Laser}; ///< Kind of reference.
    QString stageKey;             ///< Target stage's hwKey; valid iff type==Stage.
    double  fixedCm1{0.0};        ///< Fixed mixing beam (cm⁻¹); valid iff type==Fixed.
};

/*!
 * \brief One crystal/stage node. The tunable LASER source is implicit (not
 *        a \c Node); Nodes are contributed by conversion stages.
 */
struct Node {
    QString stageKey;              ///< Owning stage's hwKey (unique within a graph).
    Op      op{Op::NHG};           ///< Conversion operation.
    int     n{2};                  ///< Harmonic order for NHG (>=1); ignored otherwise.
    std::vector<InputRef> inputs;  ///< NHG -> exactly 1; SFG/DFG -> exactly 2.
    bool    isFinal{false};        ///< Marks the beam that is the LIF output axis.
};

}

/*!
 * \brief Pure value type representing an assembled LIF frequency-conversion
 *        topology, from the tunable grating fundamental through zero or
 *        more conversion nodes to the FINAL output beam.
 *
 * All values are vacuum wavenumber (cm⁻¹) end to end; convert to/from a
 * user-facing \c BC::LifConv::LaserUnit only at display call sites (see
 * lifunits.h). \c assemble() is the only validating construction path —
 * every other member reads a successfully-assembled (or default-identity)
 * instance. No \c HardwareObject or \c SettingsStorage dependency: callers
 * build the \c Node list from settings snapshots and hand it to \c
 * assemble().
 */
class LifConversion
{
public:
    /*!
     * \brief Result of \c assemble(): either a usable \c LifConversion, or
     *        a diagnostic on failure.
     *
     * Forward-declared here and defined just below the class: a nested
     * class holding a by-value \c LifConversion member cannot be defined
     * inline inside \c LifConversion's own (still-incomplete) body.
     */
    struct AssemblyResult;

    /*!
     * \brief Construct the identity conversion: output == fundamental (the
     *        zero-stage case).
     */
    LifConversion();

    /*!
     * \brief Assemble and validate a conversion graph from per-stage node
     *        descriptors.
     *
     * An empty \a nodes list yields the identity conversion (no FINAL
     * marker required). Otherwise, validation rejects the graph (returning
     * \c {false, errorString, {}}) when: any \c InputRef of type \c Stage
     * fails to resolve to a \c Node in \a nodes; a node's input count does
     * not match its \c Op (NHG=1, SFG/DFG=2); the graph does not have
     * exactly one node with \c isFinal set; the graph contains a cycle; or
     * the assembled FINAL beam has no net dependence on the tunable laser
     * source (see the comment in the .cpp on why this is the
     * currently-representable proxy for "more than one tunable source").
     *
     * A two-input stage may carry the tunable beam in either slot; \c
     * stageInput() reports whichever input tracks the fundamental, so there
     * is no requirement that the tunable beam be \c inputs[0].
     *
     * For a \c DFG the output is the difference beam \c |in0 - in1|, held as
     * a signed affine expression rather than an absolute value (so the
     * inverse and the persisted coefficients stay exact). A valid crystal
     * never operates across a zero-crossing, so the higher-frequency input is
     * fixed for the whole scan and must be wired as \c inputs[0]; the
     * difference is then the physical (non-negative) beam. \c assemble() has
     * no tuning range and cannot check this, so the construction UI warns
     * when a stage's output beam would reach zero or below over the laser's
     * range (see \c stageOutputCoeffs()).
     */
    static AssemblyResult assemble(const std::vector<BC::LifConv::Node> &nodes);

    /*!
     * \brief Return the FINAL beam wavenumber (cm⁻¹) for a given grating
     *        fundamental (cm⁻¹).
     */
    double laserToOutput(double fundamentalCm1) const;

    /*!
     * \brief Analytic inverse of \c laserToOutput: the grating fundamental
     *        (cm⁻¹) that produces a given FINAL beam wavenumber (cm⁻¹).
     *
     * Exact (the topology is affine in the fundamental); no numerics.
     */
    double outputToLaser(double outputCm1) const;

    /*!
     * \brief Return the local tunable-tracking input-beam wavenumber (cm⁻¹)
     *        seen by the node named \a stageKey, for a given grating
     *        fundamental (cm⁻¹) — what that FCU calibrates its phase-match
     *        motion against. This is whichever input carries the tunable
     *        dependence, not necessarily \c inputs[0].
     *
     * Returns \c -1.0 if \a stageKey does not name a node in this conversion.
     * A physical beam wavenumber is never negative, so this sentinel is
     * unambiguous.
     */
    double stageInput(const QString &stageKey, double fundamentalCm1) const;

    /*!
     * \brief Return the OUTPUT-beam wavenumber (cm⁻¹) produced by the node
     *        named \a stageKey, for a given grating fundamental (cm⁻¹) — the
     *        node's own conversion applied to its inputs.
     *
     * Symmetric partner to \c stageInput(): for the FINAL node this equals
     * \c laserToOutput(). Used to record each node's resolved affine
     * mapping when snapshotting the topology. Returns \c -1.0 if \a stageKey
     * does not name a node in this conversion (an unambiguous sentinel, as a
     * physical beam wavenumber is never negative).
     */
    double stageOutput(const QString &stageKey, double fundamentalCm1) const;

    /*!
     * \brief Return the affine coefficients \c {a, b} of the OUTPUT beam
     *        produced by the node named \a stageKey: wavenumber =
     *        \c a*fundamentalCm1 + b (cm⁻¹).
     *
     * These are the exact values fixed at \c assemble() time — the same ones
     * \c stageOutput() evaluates — so a caller that persists or analyzes a
     * stage's mapping reads them directly instead of reconstructing them from
     * sampled evaluations. Returns \c {0.0, -1.0} for an unknown \a stageKey;
     * that sentinel is unambiguous (a physical beam is never negative) and
     * makes \c stageOutput() return \c -1.0 at any fundamental for the same
     * key.
     */
    std::pair<double,double> stageOutputCoeffs(const QString &stageKey) const;

    /*!
     * \brief Return the affine coefficients \c {a, b} of the tunable-tracking
     *        INPUT beam seen by the node named \a stageKey (see \c
     *        stageInput()). Returns \c {0.0, -1.0} for an unknown \a stageKey.
     */
    std::pair<double,double> stageInputCoeffs(const QString &stageKey) const;

    /*!
     * \brief Return the FINAL-beam bounds (cm⁻¹) corresponding to the
     *        grating's native \a laserMinCm1 / \a laserMaxCm1, sorted
     *        ascending (the topology may reverse direction, e.g. doubling
     *        maps a max wavelength to a min wavelength).
     */
    std::pair<double,double> outputRange(double laserMinCm1, double laserMaxCm1) const;

    /*!
     * \brief Return \c true when this conversion has no stages (output ==
     *        fundamental).
     */
    bool isIdentity() const;

private:
    bool d_identity{true};   ///< \c true iff assembled from an empty node list (or default-constructed).
    BC::LifConv::detail::AffineCoeffs d_output; ///< FINAL beam coefficients vs. the fundamental.
    std::map<QString,BC::LifConv::detail::AffineCoeffs> d_primaryInput; ///< Per-stage tunable-tracking input coefficients (see stageInput()).
    std::map<QString,BC::LifConv::detail::AffineCoeffs> d_stageOutput; ///< Per-stage OUTPUT-beam coefficients.
};

struct LifConversion::AssemblyResult {
    bool ok{false};         ///< \c true iff assembly succeeded.
    QString errorString;    ///< Populated iff \c !ok.
    LifConversion conversion; ///< Valid iff \c ok.
};

#endif // LIFCONVERSION_H
