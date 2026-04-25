#include <data/loadout/loadoutmanager.h>

#include <data/loadout/chirpconfigloadout.h>
#include <data/loadout/ftmwdigitizerloadout.h>
#include <data/settings/guikeys.h>

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
                d_default =d_loadouts.constBegin().key();
        }
    }

    p_removeFromSettings(name);
    p_syncIndex();

    // If the removed loadout was the last one seeded into FtmwConfigWidget,
    // clear the stored key so the widget re-seeds from the new current loadout.
    {
        LoadoutHelper ftmwHelper({BC::Key::FtmwConfigWidget::key.toString()});
        if (ftmwHelper.get(BC::Key::FtmwConfigWidget::lastLoadout, QString()) == name)
            ftmwHelper.clearValue(BC::Key::FtmwConfigWidget::lastLoadout);
    }

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

// ── private helpers ──────────────────────────────────────────────────────────

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

    // First-run: no loadouts found → create empty "Default"
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

    if (sub.get<bool>(ftmwPresentKey)) {
        FtmwSnapshot snap;
        snap.digiHwKey = sub.get<QString>(digiHwKeyField);

        snap.rfConfig = rfConfigSnapshotFromMaps(
            sub.getGroup(rfConfigKey),
            sub.getArray(rfClocksKey));

        snap.chirpConfig = chirpConfigFromMaps(
            sub.getGroup(chirpScalarsKey),
            sub.getArray(chirpSegmentsKey),
            sub.getArray(chirpMarkersKey),
            1.0);

        snap.digitizer = ftmwDigitizerFromMaps(
            snap.digiHwKey,
            sub.getGroup(digiScalarsKey),
            sub.getArray(digiAnalogKey),
            sub.getArray(digiDigitalKey));

        loadout.ftmw = std::move(snap);
    }

    return loadout;
}

void LoadoutManager::p_writeLoadout(const HardwareLoadout &loadout)
{
    LoadoutHelper sub({key.toString(), loadout.name});
    sub.discardChanges(true);

    sub.setArray(hwMapKey, hardwareMapArray(loadout.hardwareMap));

    const bool hasFtmw = loadout.ftmw.has_value();
    sub.set(ftmwPresentKey, hasFtmw);

    if (hasFtmw) {
        const auto &snap = *loadout.ftmw;

        sub.set(digiHwKeyField, snap.digiHwKey);

        sub.setGroupValues(rfConfigKey, rfConfigScalarsMap(snap.rfConfig));
        sub.setArray(rfClocksKey, rfConfigClocksArray(snap.rfConfig));

        sub.setGroupValues(chirpScalarsKey, chirpConfigScalarsMap(snap.chirpConfig));
        sub.setArray(chirpSegmentsKey, chirpConfigSegmentsArray(snap.chirpConfig));
        sub.setArray(chirpMarkersKey, chirpConfigMarkersArray(snap.chirpConfig));

        sub.setGroupValues(digiScalarsKey, digitizerScalarsMap(snap.digitizer));
        sub.setArray(digiAnalogKey, digitizerAnalogArray(snap.digitizer));
        sub.setArray(digiDigitalKey, digitizerDigitalArray(snap.digitizer));
    }

    sub.discardChanges(false);
    sub.save();
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
