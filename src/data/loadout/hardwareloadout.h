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

// Loadout-specific clock array field keys (extends BC::Store::RFC)
namespace BC::Store::RFC {
inline constexpr QLatin1StringView clockType{"ClockType"};
inline constexpr QLatin1StringView clockOutput{"Output"};
inline constexpr QLatin1StringView clockOp{"MultOperation"};
inline constexpr QLatin1StringView clockFactor{"Factor"};
inline constexpr QLatin1StringView clockFreqMHz{"FreqMHz"};
inline constexpr QLatin1StringView hwKey{"HwKey"};        // shared by clock and hardware-map arrays
inline constexpr QLatin1StringView hwImpl{"Implementation"};
}

struct FtmwPreset {
    RfConfigSnapshot rfConfig;
    ChirpConfig chirpConfig;
    FtmwDigitizerConfig digitizer{""};
    QString digiHwKey;
    QDateTime lastModified;
};

struct HardwareLoadout {
    QString name;
    std::map<QString, QString, std::less<>> hardwareMap;
    std::map<QString, FtmwPreset, std::less<>> ftmwPresets;
    QString currentFtmwPresetName;
    QDateTime lastModified;
};

namespace BC::Loadout {

using Map  = SettingsStorage::SettingsMap;
using Maps = std::vector<SettingsStorage::SettingsMap>;

// RfConfigSnapshot ↔ maps
Map  rfConfigScalarsMap(const RfConfigSnapshot &snap);
Maps rfConfigClocksArray(const RfConfigSnapshot &snap);
RfConfigSnapshot rfConfigSnapshotFromMaps(const Map &scalars, const Maps &clocks);

// Per-component copy helpers used by FtmwConfigDialog tabs
void copyClocksMatching(const RfConfigSnapshot &source,
                        RfConfigSnapshot &dest,
                        const std::set<QString> &allowedHwKeys);

void copyRfScalars(const RfConfigSnapshot &source, RfConfigSnapshot &dest);

// Hardware map ↔ array  (used by LoadoutManager for QSettings persistence)
Maps hardwareMapArray(const std::map<QString, QString, std::less<>> &hwMap);
std::map<QString, QString, std::less<>> hardwareMapFromArray(const Maps &array);

} // namespace BC::Loadout

#endif // BC_HARDWARELOADOUT_H