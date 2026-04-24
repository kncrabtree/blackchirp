#ifndef BC_LOADOUT_CHIRPCONFIGLOADOUT_H
#define BC_LOADOUT_CHIRPCONFIGLOADOUT_H

#include <data/storage/settingsstorage.h>
#include <data/experiment/chirpconfig.h>

namespace BC::Loadout {

SettingsStorage::SettingsMap chirpConfigScalarsMap(const ChirpConfig &cc);

std::vector<SettingsStorage::SettingsMap> chirpConfigSegmentsArray(const ChirpConfig &cc);

std::vector<SettingsStorage::SettingsMap> chirpConfigMarkersArray(const ChirpConfig &cc);

ChirpConfig chirpConfigFromMaps(const SettingsStorage::SettingsMap &scalars,
                                const std::vector<SettingsStorage::SettingsMap> &segments,
                                const std::vector<SettingsStorage::SettingsMap> &markers,
                                double awgSampleRateSps);

} // namespace BC::Loadout

#endif