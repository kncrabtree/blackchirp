#ifndef RFCONFIGSNAPSHOT_H
#define RFCONFIGSNAPSHOT_H

#include <QHash>

#include <data/experiment/rfconfig.h>

/// \brief Serializable snapshot of the persistable fields of an RfConfig.
///
/// `RfConfigSnapshot` captures only the subset of `RfConfig` state that
/// belongs to a stored FTMW preset: the up- and down-conversion mixer
/// settings, the AWG and chirp multipliers, and the desired clock
/// frequencies for each clock role. Hardware identity (which physical
/// clock implements which role) is recorded separately in the loadout's
/// hardware map and is not part of the snapshot.
///
/// `FtmwPreset` owns one `RfConfigSnapshot`. The loadout persistence
/// layer converts snapshots to and from `SettingsStorage::SettingsMap`
/// records via the `BC::Loadout::rfConfigScalarsMap`,
/// `BC::Loadout::rfConfigClocksArray`, and
/// `BC::Loadout::rfConfigSnapshotFromMaps` helpers.
struct RfConfigSnapshot
{
    /// \brief Whether the upconversion and downconversion local oscillators share the same clock source.
    bool commonUpDownLO{false};
    /// \brief AWG output multiplier applied before the upconversion mixer.
    double awgMult{1.0};
    /// \brief Sideband selection for the upconversion mixer.
    RfConfig::Sideband upMixSideband{RfConfig::UpperSideband};
    /// \brief Frequency multiplier applied to the chirp before downconversion.
    double chirpMult{1.0};
    /// \brief Sideband selection for the downconversion mixer.
    RfConfig::Sideband downMixSideband{RfConfig::UpperSideband};
    /// \brief Desired frequency for each populated clock role.
    QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks;

    /// \brief Build a snapshot from the persistable fields of `c`.
    static RfConfigSnapshot fromRfConfig(const RfConfig &c);
    /// \brief Apply the snapshot's fields to `c`, leaving non-persisted state untouched.
    void applyTo(RfConfig &c) const;
};

#endif // RFCONFIGSNAPSHOT_H
