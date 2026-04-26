#ifndef BC_LOADOUTMANAGER_H
#define BC_LOADOUTMANAGER_H

#include <optional>

#include <QHash>
#include <QMutex>
#include <QObject>
#include <QString>
#include <QStringList>

#include <data/loadout/hardwareloadout.h>
#include <data/storage/settingsstorage.h>

namespace BC::Store::LM {
inline constexpr QLatin1StringView key{"Loadouts"};
inline constexpr QLatin1StringView current{"currentLoadout"};
inline constexpr QLatin1StringView defaultName{"defaultLoadout"};
inline constexpr QLatin1StringView namesKey{"names"};
inline constexpr QLatin1StringView nameField{"name"};
inline constexpr QLatin1StringView hwMapKey{"hardwareMap"};
inline constexpr QLatin1StringView digiHwKeyField{"DigiHwKey"};
inline constexpr QLatin1StringView rfScalarsKey{"rfScalars"};
inline constexpr QLatin1StringView rfClocksKey{"rfClocks"};
inline constexpr QLatin1StringView chirpScalarsKey{"chirpScalars"};
inline constexpr QLatin1StringView chirpSegmentsKey{"chirpSegments"};
inline constexpr QLatin1StringView chirpMarkersKey{"chirpMarkers"};
inline constexpr QLatin1StringView digiScalarsKey{"digiScalars"};
inline constexpr QLatin1StringView digiAnalogKey{"digiAnalog"};
inline constexpr QLatin1StringView digiDigitalKey{"digiDigital"};
inline constexpr QLatin1StringView ftmwPresetsKey{"ftmwPresets"};
inline constexpr QLatin1StringView ftmwPresetNamesKey{"ftmwPresetNames"};
inline constexpr QLatin1StringView defaultFtmwPresetKey{"defaultFtmwPreset"};
inline constexpr QLatin1StringView currentFtmwPresetKey{"currentFtmwPreset"};
inline constexpr QLatin1StringView lastModifiedKey{"lastModified"};
inline constexpr QLatin1StringView lastUsedFtmwPresetName{"__LastUsed__"};
}

class LoadoutManagerTest;

class LoadoutManager : public QObject, public SettingsStorage
{
    Q_OBJECT

public:
    static LoadoutManager &instance();
    ~LoadoutManager() override;

    // Loadout CRUD
    QStringList loadoutNames() const;
    bool loadoutExists(const QString &name) const;
    std::optional<HardwareLoadout> getLoadout(const QString &name) const;
    bool putLoadout(const HardwareLoadout &loadout);
    bool removeLoadout(const QString &name);

    // Current/default loadout
    QString currentLoadoutName() const;
    void setCurrentLoadoutName(const QString &name);
    std::optional<HardwareLoadout> currentLoadout() const;

    QString defaultLoadoutName() const;
    void setDefaultLoadoutName(const QString &name);
    std::optional<HardwareLoadout> defaultLoadout() const;

    QStringList loadoutsMatchingHwKey(const QString &hwKey) const;

    // FTMW preset CRUD
    std::optional<FtmwPreset> getFtmwPreset(const QString &loadoutName, const QString &presetName) const;
    bool putFtmwPreset(const QString &loadoutName, const QString &presetName, const FtmwPreset &preset);
    bool removeFtmwPreset(const QString &loadoutName, const QString &presetName);
    bool renameFtmwPreset(const QString &loadoutName, const QString &oldName, const QString &newName);
    bool ftmwPresetExists(const QString &loadoutName, const QString &presetName) const;
    QStringList ftmwPresetNames(const QString &loadoutName, bool includeLastUsed = false) const;
    bool clearFtmwPresets(const QString &loadoutName);

    // Current/default FTMW preset
    QString currentFtmwPresetName(const QString &loadoutName) const;
    bool setCurrentFtmwPresetName(const QString &loadoutName, const QString &presetName);
    QString defaultFtmwPresetName(const QString &loadoutName) const;
    bool setDefaultFtmwPresetName(const QString &loadoutName, const QString &presetName);
    std::optional<FtmwPreset> currentFtmwPreset(const QString &loadoutName) const;

signals:
    void loadoutAdded(QString name);
    void loadoutRemoved(QString name);
    void loadoutChanged(QString name);
    void currentLoadoutChanged(QString name);
    void defaultLoadoutChanged(QString name);

    void ftmwPresetAdded(QString loadoutName, QString presetName);
    void ftmwPresetRemoved(QString loadoutName, QString presetName);
    void ftmwPresetChanged(QString loadoutName, QString presetName);
    void currentFtmwPresetChanged(QString loadoutName, QString presetName);
    void defaultFtmwPresetChanged(QString loadoutName, QString presetName);

private:
    LoadoutManager();
    LoadoutManager(QAnyStringView orgName, QAnyStringView appName);

    class LoadoutHelper : public SettingsStorage
    {
        friend class LoadoutManager;
    public:
        LoadoutHelper(const QStringList &keys) : SettingsStorage(keys) {}
    };

    void p_loadAll();
    HardwareLoadout p_readLoadout(const QString &name) const;
    void p_writeLoadout(const HardwareLoadout &loadout);
    void p_removeFromSettings(const QString &name);
    void p_syncIndex();

    FtmwPreset p_readFtmwPreset(const QString &loadoutName, const QString &presetName) const;
    void p_writeFtmwPreset(const QString &loadoutName, const QString &presetName, const FtmwPreset &preset);
    void p_removeFtmwPresetFromSettings(const QString &loadoutName, const QString &presetName);
    void p_syncFtmwPresetIndex(const QString &loadoutName);
    void p_writeFtmwPresetPointers(const QString &loadoutName);

    QHash<QString, HardwareLoadout> d_loadouts;
    QString d_current;
    QString d_default;
    mutable QMutex d_mutex;

    static LoadoutManager *s_instance;
    friend class LoadoutManagerTest;
};

#endif // BC_LOADOUTMANAGER_H
