#include <data/loadout/loadoutmanager.h>

#include <data/loadout/chirpconfigloadout.h>
#include <data/loadout/ftmwdigitizerloadout.h>
#include <data/settings/hardwarekeys.h>

using namespace BC::Store::LM;
using namespace BC::Loadout;
using namespace Qt::StringLiterals;

LoadoutManager *LoadoutManager::s_instance = nullptr;

LoadoutManager &LoadoutManager::instance()
{
    if (!s_instance)
        s_instance = new LoadoutManager;
    return *s_instance;
}

LoadoutManager::LoadoutManager()
    : QObject(), SettingsStorage(QStringList{key.toString()})
{
    p_loadAll();
}

LoadoutManager::LoadoutManager(QAnyStringView orgName, QAnyStringView appName)
    : QObject(), SettingsStorage(orgName, appName, {key.toString()})
{
    p_loadAll();
}

LoadoutManager::~LoadoutManager()
{
}

QStringList LoadoutManager::loadoutNames() const
{
    QMutexLocker lk(&d_mutex);
    return QStringList(d_loadouts.keyBegin(), d_loadouts.keyEnd());
}

bool LoadoutManager::loadoutExists(const QString &name) const
{
    QMutexLocker lk(&d_mutex);
    return d_loadouts.contains(name);
}

std::optional<HardwareLoadout> LoadoutManager::getLoadout(const QString &name) const
{
    QMutexLocker lk(&d_mutex);
    auto it = d_loadouts.find(name);
    if (it == d_loadouts.end())
        return std::nullopt;
    return *it;
}

bool LoadoutManager::putLoadout(const HardwareLoadout &loadout)
{
    if (loadout.name.isEmpty())
        return false;

    bool isNew = false;
    {
        QMutexLocker lk(&d_mutex);
        isNew = !d_loadouts.contains(loadout.name);
        d_loadouts.insert(loadout.name, loadout);
    }

    p_writeLoadout(loadout);
    p_syncIndex();

    if (isNew)
        emit loadoutAdded(loadout.name);
    else
        emit loadoutChanged(loadout.name);

    return true;
}

bool LoadoutManager::removeLoadout(const QString &name)
{
    {
        QMutexLocker lk(&d_mutex);
        if (!d_loadouts.remove(name))
            return false;

        if (d_current == name) {
            if (!d_loadouts.isEmpty())
            {
                if(d_default != name)
                    d_current = d_default;
                else
                    d_current = d_loadouts.constBegin().key();
            }
            else
                d_current.clear();
        }
        if (d_default == name) {
            if(d_loadouts.isEmpty())
                d_default.clear();
            else
                d_default = d_loadouts.constBegin().key();
        }
    }

    p_removeFromSettings(name);
    p_syncIndex();

    {
        QMutexLocker lk(&d_mutex);
        set(current, d_current, true);
        set(defaultName, d_default, true);
    }

    emit loadoutRemoved(name);
    return true;
}

QString LoadoutManager::currentLoadoutName() const
{
    QMutexLocker lk(&d_mutex);
    return d_current;
}

void LoadoutManager::setCurrentLoadoutName(const QString &name)
{
    {
        QMutexLocker lk(&d_mutex);
        if (d_current == name)
            return;
        d_current = name;
        set(current, name, true);
    }
    emit currentLoadoutChanged(name);
}

std::optional<HardwareLoadout> LoadoutManager::currentLoadout() const
{
    return getLoadout(currentLoadoutName());
}

QString LoadoutManager::defaultLoadoutName() const
{
    QMutexLocker lk(&d_mutex);
    return d_default;
}

void LoadoutManager::setDefaultLoadoutName(const QString &name)
{
    {
        QMutexLocker lk(&d_mutex);
        if (d_default == name)
            return;
        d_default = name;
        set(defaultName, name, true);
    }
    emit defaultLoadoutChanged(name);
}

std::optional<HardwareLoadout> LoadoutManager::defaultLoadout() const
{
    return getLoadout(defaultLoadoutName());
}

QStringList LoadoutManager::loadoutsMatchingHwKey(const QString &hwKey) const
{
    QMutexLocker lk(&d_mutex);
    QStringList result;
    for (auto it = d_loadouts.constBegin(); it != d_loadouts.constEnd(); ++it) {
        if (it.value().hardwareMap.count(hwKey))
            result.append(it.key());
    }
    return result;
}

// ── FTMW preset CRUD ─────────────────────────────────────────────────────────

std::optional<FtmwPreset> LoadoutManager::getFtmwPreset(const QString &loadoutName, const QString &presetName) const
{
    QMutexLocker lk(&d_mutex);
    auto it = d_loadouts.find(loadoutName);
    if (it == d_loadouts.end())
        return std::nullopt;
    auto pit = it->ftmwPresets.find(presetName);
    if (pit == it->ftmwPresets.end())
        return std::nullopt;
    return pit->second;
}

bool LoadoutManager::putFtmwPreset(const QString &loadoutName, const QString &presetName, const FtmwPreset &preset)
{
    if (loadoutName.isEmpty() || presetName.isEmpty())
        return false;

    FtmwPreset stamped = preset;
    stamped.lastModified = QDateTime::currentDateTimeUtc();

    bool isNew = false;
    {
        QMutexLocker lk(&d_mutex);
        auto it = d_loadouts.find(loadoutName);
        if (it == d_loadouts.end())
            return false;
        isNew = !it->ftmwPresets.count(presetName);
        it->ftmwPresets[presetName] = stamped;
    }

    p_writeFtmwPreset(loadoutName, presetName, stamped);
    p_syncFtmwPresetIndex(loadoutName);

    if (isNew)
        emit ftmwPresetAdded(loadoutName, presetName);
    else
        emit ftmwPresetChanged(loadoutName, presetName);

    return true;
}

bool LoadoutManager::removeFtmwPreset(const QString &loadoutName, const QString &presetName)
{
    if (loadoutName.isEmpty() || presetName.isEmpty())
        return false;

    {
        QMutexLocker lk(&d_mutex);
        auto it = d_loadouts.find(loadoutName);
        if (it == d_loadouts.end())
            return false;
        if (!it->ftmwPresets.count(presetName))
            return false;
        if (it->currentFtmwPresetName == presetName)
            return false;  // active preset cannot be removed

        it->ftmwPresets.erase(presetName);
    }

    p_removeFtmwPresetFromSettings(loadoutName, presetName);
    p_writeFtmwPresetPointers(loadoutName);
    p_syncFtmwPresetIndex(loadoutName);

    emit ftmwPresetRemoved(loadoutName, presetName);
    return true;
}

bool LoadoutManager::renameFtmwPreset(const QString &loadoutName, const QString &oldName, const QString &newName)
{
    if (loadoutName.isEmpty() || oldName.isEmpty() || newName.isEmpty())
        return false;
    if (oldName == lastUsedFtmwPresetName || newName == lastUsedFtmwPresetName)
        return false;
    if (oldName == newName)
        return true;

    FtmwPreset movedPreset;
    bool currentChanged = false;

    {
        QMutexLocker lk(&d_mutex);
        auto it = d_loadouts.find(loadoutName);
        if (it == d_loadouts.end())
            return false;
        auto pit = it->ftmwPresets.find(oldName);
        if (pit == it->ftmwPresets.end())
            return false;
        if (it->ftmwPresets.count(newName))
            return false;  // duplicate

        movedPreset = std::move(pit->second);
        it->ftmwPresets.erase(pit);
        it->ftmwPresets[newName] = movedPreset;

        if (it->currentFtmwPresetName == oldName) {
            it->currentFtmwPresetName = newName;
            currentChanged = true;
        }
    }

    p_removeFtmwPresetFromSettings(loadoutName, oldName);
    p_writeFtmwPreset(loadoutName, newName, movedPreset);
    p_writeFtmwPresetPointers(loadoutName);
    p_syncFtmwPresetIndex(loadoutName);

    emit ftmwPresetRemoved(loadoutName, oldName);
    emit ftmwPresetAdded(loadoutName, newName);
    if (currentChanged)
        emit currentFtmwPresetChanged(loadoutName, newName);

    return true;
}

bool LoadoutManager::ftmwPresetExists(const QString &loadoutName, const QString &presetName) const
{
    QMutexLocker lk(&d_mutex);
    auto it = d_loadouts.find(loadoutName);
    if (it == d_loadouts.end())
        return false;
    return it->ftmwPresets.count(presetName) > 0;
}

QStringList LoadoutManager::ftmwPresetNames(const QString &loadoutName, bool includeLastUsed) const
{
    QMutexLocker lk(&d_mutex);
    auto it = d_loadouts.find(loadoutName);
    if (it == d_loadouts.end())
        return {};

    QStringList names;
    for (const auto &[name, preset] : it->ftmwPresets) {
        if (!includeLastUsed && name == lastUsedFtmwPresetName)
            continue;
        names.append(name);
    }
    return names;
}

bool LoadoutManager::clearFtmwPresets(const QString &loadoutName)
{
    {
        QMutexLocker lk(&d_mutex);
        auto it = d_loadouts.find(loadoutName);
        if (it == d_loadouts.end())
            return false;
        it->ftmwPresets.clear();
        it->currentFtmwPresetName.clear();
    }

    SettingsStorage::purgeGroup({key.toString(), loadoutName, ftmwPresetsKey.toString()});
    p_writeFtmwPresetPointers(loadoutName);
    p_syncFtmwPresetIndex(loadoutName);

    emit loadoutChanged(loadoutName);
    return true;
}

// ── FTMW preset current/default ───────────────────────────────────────────────

QString LoadoutManager::currentFtmwPresetName(const QString &loadoutName) const
{
    QMutexLocker lk(&d_mutex);
    auto it = d_loadouts.find(loadoutName);
    if (it == d_loadouts.end())
        return {};
    return it->currentFtmwPresetName;
}

bool LoadoutManager::setCurrentFtmwPresetName(const QString &loadoutName, const QString &presetName)
{
    {
        QMutexLocker lk(&d_mutex);
        auto it = d_loadouts.find(loadoutName);
        if (it == d_loadouts.end())
            return false;

        // Allow empty, __LastUsed__, or an existing named preset
        if (!presetName.isEmpty() &&
            presetName != lastUsedFtmwPresetName &&
            !it->ftmwPresets.count(presetName))
            return false;

        if (it->currentFtmwPresetName == presetName)
            return true;

        it->currentFtmwPresetName = presetName;
    }

    p_writeFtmwPresetPointers(loadoutName);
    emit currentFtmwPresetChanged(loadoutName, presetName);
    return true;
}


std::optional<FtmwPreset> LoadoutManager::currentFtmwPreset(const QString &loadoutName) const
{
    QMutexLocker lk(&d_mutex);
    auto it = d_loadouts.find(loadoutName);
    if (it == d_loadouts.end())
        return std::nullopt;

    if (!it->currentFtmwPresetName.isEmpty()) {
        auto pit = it->ftmwPresets.find(it->currentFtmwPresetName);
        if (pit != it->ftmwPresets.end())
            return pit->second;
    }

    return std::nullopt;
}

// ── private helpers ───────────────────────────────────────────────────────────

void LoadoutManager::p_loadAll()
{
    d_current = get<QString>(current);
    d_default = get<QString>(defaultName);

    const auto namesMaps = getArray(namesKey);
    for (const auto &m : namesMaps) {
        auto it = m.find(nameField);
        if (it == m.end())
            continue;
        const QString n = it->second.value<QString>();
        if (!n.isEmpty())
            d_loadouts.insert(n, p_readLoadout(n));
    }

    if (d_loadouts.isEmpty()) {
        HardwareLoadout def;
        def.name = u"Default"_s;
        d_loadouts.insert(def.name, def);
        d_current = def.name;
        d_default = def.name;
        set(current, d_current);
        set(defaultName, d_default);
        p_writeLoadout(def);
        p_syncIndex();
        save();
    }
}

HardwareLoadout LoadoutManager::p_readLoadout(const QString &name) const
{
    LoadoutHelper sub({key.toString(), name});
    sub.discardChanges(true);

    HardwareLoadout loadout;
    loadout.name = name;
    loadout.hardwareMap = hardwareMapFromArray(sub.getArray(hwMapKey));
    loadout.currentFtmwPresetName = sub.get<QString>(currentFtmwPresetKey);

    const auto lastModStr = sub.get<QString>(lastModifiedKey);
    if (!lastModStr.isEmpty())
        loadout.lastModified = QDateTime::fromString(lastModStr, Qt::ISODate);

    const auto presetNamesMaps = sub.getArray(ftmwPresetNamesKey);
    for (const auto &m : presetNamesMaps) {
        auto it = m.find(nameField);
        if (it == m.end())
            continue;
        const QString pName = it->second.value<QString>();
        if (!pName.isEmpty())
            loadout.ftmwPresets[pName] = p_readFtmwPreset(name, pName);
    }

    return loadout;
}

void LoadoutManager::p_writeLoadout(const HardwareLoadout &loadout)
{
    LoadoutHelper sub({key.toString(), loadout.name});
    sub.discardChanges(true);

    sub.setArray(hwMapKey, hardwareMapArray(loadout.hardwareMap));
    sub.set(currentFtmwPresetKey, loadout.currentFtmwPresetName);
    sub.set(lastModifiedKey, loadout.lastModified.isValid()
            ? loadout.lastModified.toString(Qt::ISODate)
            : QString{});

    Maps names;
    for (const auto &[pName, preset] : loadout.ftmwPresets) {
        Map m;
        m[nameField] = pName;
        names.push_back(std::move(m));
    }
    sub.setArray(ftmwPresetNamesKey, names);

    sub.discardChanges(false);
    sub.save();

    for (const auto &[pName, preset] : loadout.ftmwPresets)
        p_writeFtmwPreset(loadout.name, pName, preset);
}

void LoadoutManager::p_removeFromSettings(const QString &name)
{
    SettingsStorage::purgeGroup({key.toString(), name});
}

void LoadoutManager::p_syncIndex()
{
    QMutexLocker lk(&d_mutex);
    Maps names;
    for (auto it = d_loadouts.constBegin(); it != d_loadouts.constEnd(); ++it) {
        Map m;
        m[nameField] = it.key();
        names.push_back(std::move(m));
    }
    setArray(namesKey, names, true);
}

FtmwPreset LoadoutManager::p_readFtmwPreset(const QString &loadoutName, const QString &presetName) const
{
    LoadoutHelper sub({key.toString(), loadoutName, ftmwPresetsKey.toString(), presetName});
    sub.discardChanges(true);

    FtmwPreset preset;
    preset.digiHwKey = sub.get<QString>(digiHwKeyField);

    const auto lastModStr = sub.get<QString>(lastModifiedKey);
    if (!lastModStr.isEmpty())
        preset.lastModified = QDateTime::fromString(lastModStr, Qt::ISODate);

    preset.rfConfig = rfConfigSnapshotFromMaps(
        sub.getGroup(rfScalarsKey),
        sub.getArray(rfClocksKey));

    preset.chirpConfig = chirpConfigFromMaps(
        sub.getGroup(chirpScalarsKey),
        sub.getArray(chirpSegmentsKey),
        sub.getArray(chirpMarkersKey),
        1.0);

    preset.digitizer = ftmwDigitizerFromMaps(
        preset.digiHwKey,
        sub.getGroup(digiScalarsKey),
        sub.getArray(digiAnalogKey),
        sub.getArray(digiDigitalKey));

    return preset;
}

void LoadoutManager::p_writeFtmwPreset(const QString &loadoutName, const QString &presetName, const FtmwPreset &preset)
{
    LoadoutHelper sub({key.toString(), loadoutName, ftmwPresetsKey.toString(), presetName});
    sub.discardChanges(true);

    sub.set(digiHwKeyField, preset.digiHwKey);
    sub.set(lastModifiedKey, preset.lastModified.isValid()
            ? preset.lastModified.toString(Qt::ISODate)
            : QString{});

    sub.setGroupValues(rfScalarsKey, rfConfigScalarsMap(preset.rfConfig));
    sub.setArray(rfClocksKey, rfConfigClocksArray(preset.rfConfig));

    sub.setGroupValues(chirpScalarsKey, chirpConfigScalarsMap(preset.chirpConfig));
    sub.setArray(chirpSegmentsKey, chirpConfigSegmentsArray(preset.chirpConfig));
    sub.setArray(chirpMarkersKey, chirpConfigMarkersArray(preset.chirpConfig));

    sub.setGroupValues(digiScalarsKey, digitizerScalarsMap(preset.digitizer));
    sub.setArray(digiAnalogKey, digitizerAnalogArray(preset.digitizer));
    sub.setArray(digiDigitalKey, digitizerDigitalArray(preset.digitizer));

    sub.discardChanges(false);
    sub.save();
}

void LoadoutManager::p_removeFtmwPresetFromSettings(const QString &loadoutName, const QString &presetName)
{
    SettingsStorage::purgeGroup({key.toString(), loadoutName, ftmwPresetsKey.toString(), presetName});
}

void LoadoutManager::p_syncFtmwPresetIndex(const QString &loadoutName)
{
    Maps names;
    {
        QMutexLocker lk(&d_mutex);
        auto it = d_loadouts.find(loadoutName);
        if (it == d_loadouts.end())
            return;
        for (const auto &[pName, preset] : it->ftmwPresets) {
            Map m;
            m[nameField] = pName;
            names.push_back(std::move(m));
        }
    }

    LoadoutHelper sub({key.toString(), loadoutName});
    sub.discardChanges(true);
    sub.setArray(ftmwPresetNamesKey, names);
    sub.discardChanges(false);
    sub.save();
}

void LoadoutManager::p_writeFtmwPresetPointers(const QString &loadoutName)
{
    QString cur;
    {
        QMutexLocker lk(&d_mutex);
        auto it = d_loadouts.find(loadoutName);
        if (it == d_loadouts.end())
            return;
        cur = it->currentFtmwPresetName;
    }

    LoadoutHelper sub({key.toString(), loadoutName});
    sub.discardChanges(true);
    sub.set(currentFtmwPresetKey, cur);
    sub.discardChanges(false);
    sub.save();
}
