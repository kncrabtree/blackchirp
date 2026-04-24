#ifndef BC_LOADOUT_FTMWDIGITIZERLOADOUT_H
#define BC_LOADOUT_FTMWDIGITIZERLOADOUT_H

#include <data/storage/settingsstorage.h>
#include <data/experiment/hardware/core/ftmwdigitizerconfig.h>

namespace BC::Loadout {

SettingsStorage::SettingsMap digitizerScalarsMap(const FtmwDigitizerConfig &cfg);

std::vector<SettingsStorage::SettingsMap> digitizerAnalogArray(const FtmwDigitizerConfig &cfg);

std::vector<SettingsStorage::SettingsMap> digitizerDigitalArray(const FtmwDigitizerConfig &cfg);

FtmwDigitizerConfig ftmwDigitizerFromMaps(const QString &hwKey,
                                          const SettingsStorage::SettingsMap &scalars,
                                          const std::vector<SettingsStorage::SettingsMap> &analog,
                                          const std::vector<SettingsStorage::SettingsMap> &digital);

} // namespace BC::Loadout

#endif // BC_LOADOUT_FTMWDIGITIZERLOADOUT_H
