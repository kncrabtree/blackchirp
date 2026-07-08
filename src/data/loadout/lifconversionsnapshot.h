#ifndef LIFCONVERSIONSNAPSHOT_H
#define LIFCONVERSIONSNAPSHOT_H

#include <functional>
#include <vector>

#include <QString>

#include <data/lif/lifconversion.h>

namespace BC::LifConv {

/// \brief Per-stage wiring captured by a LifConversionSnapshot.
///
/// The input references and FINAL marker for one conversion node, keyed by
/// the stage's hwKey. Excludes \c op/\c n: those are hardware identity /
/// registered settings (see LifFreqConversionStage::conversionOp() and
/// harmonicOrder()), not per-experiment wiring.
struct StageWiring {
    QString stageKey;
    std::vector<InputRef> inputs;  ///< NHG -> 1 entry; SFG/DFG -> 2 entries.
    bool    isFinal{false};
};

}

/*!
 * \brief Serializable snapshot of the persistable wiring of a LIF
 *        frequency-conversion topology.
 *
 * Mirrors RfConfigSnapshot: captures only the per-experiment-editable
 * subset of a \c BC::LifConv::Node list — the input wiring and FINAL
 * marker for each stage — plus the hwKey of the active \c LifLaser the
 * wiring was captured against. \c op/harmonic order are hardware-owned and
 * are not part of the snapshot; \c toNodes() joins them back in via
 * caller-supplied callbacks (in practice backed by per-stage
 * \c SettingsStorage hardware snapshots).
 *
 * \c laserKey is provenance only: it records which laser the wiring was
 * captured against. Applying a preset should substitute the *current*
 * active laser key rather than trusting the stored one.
 */
struct LifConversionSnapshot
{
    std::vector<BC::LifConv::StageWiring> wiring;
    QString laserKey;

    /// \brief Build a snapshot from the wiring/isFinal fields of \a nodes.
    static LifConversionSnapshot fromNodes(const std::vector<BC::LifConv::Node> &nodes,
                                           const QString &laserKey);

    /*!
     * \brief Rebuild a node list from this snapshot's wiring, joining in
     *        \c op/\c n via the supplied callbacks.
     * \param opOf Returns the conversion operation for a given stageKey.
     * \param harmonicOf Returns the harmonic order for a given stageKey.
     */
    std::vector<BC::LifConv::Node> toNodes(
        const std::function<BC::LifConv::Op(const QString&)> &opOf,
        const std::function<int(const QString&)> &harmonicOf) const;
};

#endif // LIFCONVERSIONSNAPSHOT_H
