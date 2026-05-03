#ifndef BC_HARDWARELOADOUT_H
#define BC_HARDWARELOADOUT_H

#include <map>
#include <set>
#include <vector>

#include <QDateTime>
#include <QString>

#include <data/experiment/rfconfig.h>
#include <data/loadout/rfconfigsnapshot.h>
#include <data/loadout/chirpconfigloadout.h>
#include <data/loadout/ftmwdigitizerloadout.h>

/// \brief Loadout-specific QSettings field keys that extend the `BC::Store::RFC` namespace.
///
/// These keys name the sub-fields of a stored RF clock entry and of a
/// hardware-map array record inside a `LoadoutManager` settings tree.
/// They are declared here, alongside the loadout data model, to keep the
/// persistence vocabulary close to the structs it serializes.
namespace BC::Store::RFC {
/// \brief Clock role identifier (e.g., upconversion LO, reference).
inline constexpr QLatin1StringView clockType{"ClockType"};
/// \brief Output port label for the clock.
inline constexpr QLatin1StringView clockOutput{"Output"};
/// \brief Multiplication or division operation applied to the clock signal.
inline constexpr QLatin1StringView clockOp{"MultOperation"};
/// \brief Numeric multiplier or divisor applied to the clock signal.
inline constexpr QLatin1StringView clockFactor{"Factor"};
/// \brief Desired clock output frequency in MHz.
inline constexpr QLatin1StringView clockFreqMHz{"FreqMHz"};
/// \brief Profile identity ("<Type>.<label>", shared by clock and hardware-map entries).
inline constexpr QLatin1StringView hwKey{"HwKey"};
/// \brief Implementation key carried by the referenced profile when the loadout was last saved.
inline constexpr QLatin1StringView hwImpl{"Implementation"};
}

/// \brief Named FTMW operating point owned by a `HardwareLoadout`.
///
/// An `FtmwPreset` aggregates the four pieces of state that fully
/// describe an FTMW measurement configuration: the RF chain
/// (`RfConfigSnapshot`), the chirp waveform (`ChirpConfig`), the
/// digitizer settings (`FtmwDigitizerConfig`), and the hardware key of
/// the digitizer the settings were captured from. Presets cannot exist
/// outside a loadout; their lifetime is managed entirely by
/// `LoadoutManager`.
struct FtmwPreset {
    /// \brief Persistable RF-chain state for the preset.
    RfConfigSnapshot rfConfig;
    /// \brief Chirp waveform definition.
    ChirpConfig chirpConfig;
    /// \brief Digitizer configuration captured from the digitizer named by `digiHwKey`.
    FtmwDigitizerConfig digitizer{""};
    /// \brief Hardware key of the digitizer profile this preset was captured from.
    QString digiHwKey;
    /// \brief Timestamp of the most recent write to this preset.
    QDateTime lastModified;
};

/// \brief Named set of member profiles plus the FTMW presets it owns.
///
/// A `HardwareLoadout` records the profile identities (`"<Type>.<label>"`)
/// that make up a complete hardware configuration, alongside the
/// implementation key each member profile carried at the time the loadout
/// was last saved (a denormalized field, used for validation and drift
/// detection — the canonical implementation lives on the profile in
/// `HardwareProfileManager`). It also holds the named `FtmwPreset`
/// operating points associated with that configuration. `LoadoutManager`
/// owns the persistent collection of loadouts; instances are passed
/// around by value and serialized into a QSettings subtree on write.
struct HardwareLoadout {
    /// \brief User-visible loadout name (also the QSettings subgroup key).
    QString name;
    /// \brief Member profile identities (`"<Type>.<label>"`) and the implementation each profile carried at save time.
    std::map<QString, QString, std::less<>> hardwareMap;
    /// \brief Named FTMW presets owned by this loadout, including the `__LastUsed__` sentinel when present.
    std::map<QString, FtmwPreset, std::less<>> ftmwPresets;
    /// \brief Name of the preset that drives initial widget population for this loadout.
    QString currentFtmwPresetName;
    /// \brief Timestamp of the most recent write to this loadout.
    QDateTime lastModified;
};

/// \brief Free-function helpers that convert loadout structs to and from `SettingsStorage::SettingsMap` records.
///
/// `LoadoutManager` uses these helpers to flatten `HardwareLoadout` and
/// `FtmwPreset` instances into the scalar/array QSettings layout
/// described in :doc:`/user_guide/hardware_config/loadouts` and to
/// reconstruct them on read. The `copyClocksMatching` and `copyRfScalars`
/// helpers support the per-component copy operations exposed by the
/// FTMW configuration dialog tabs.
namespace BC::Loadout {

using Map  = SettingsStorage::SettingsMap;
using Maps = std::vector<SettingsStorage::SettingsMap>;

/// \brief Flatten an `RfConfigSnapshot` into the scalar fields persisted under a preset's `rfScalars` group.
Map  rfConfigScalarsMap(const RfConfigSnapshot &snap);
/// \brief Flatten an `RfConfigSnapshot`'s clock table into the array persisted under a preset's `rfClocks` group.
Maps rfConfigClocksArray(const RfConfigSnapshot &snap);
/// \brief Reconstruct an `RfConfigSnapshot` from the `scalars` map and `clocks` array read out of QSettings.
RfConfigSnapshot rfConfigSnapshotFromMaps(const Map &scalars, const Maps &clocks);

/// \brief Copy clock entries from `source` to `dest` whose hardware key is in `allowedHwKeys`.
void copyClocksMatching(const RfConfigSnapshot &source,
                        RfConfigSnapshot &dest,
                        const std::set<QString> &allowedHwKeys);

/// \brief Copy the scalar (non-clock) RF-chain fields from `source` to `dest`.
void copyRfScalars(const RfConfigSnapshot &source, RfConfigSnapshot &dest);

/// \brief Flatten a hardware map into the array persisted under a loadout's `hardwareMap` group.
Maps hardwareMapArray(const std::map<QString, QString, std::less<>> &hwMap);
/// \brief Reconstruct a hardware map from the array read out of QSettings.
std::map<QString, QString, std::less<>> hardwareMapFromArray(const Maps &array);

} // namespace BC::Loadout

#endif // BC_HARDWARELOADOUT_H
