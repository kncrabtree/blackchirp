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

/// \brief QSettings field keys for the `LoadoutManager` storage tree.
///
/// All loadout state is persisted under the top-level `Loadouts/` group
/// named by `key`. The remaining identifiers name the scalar fields and
/// array sub-groups used by `LoadoutManager` and the
/// `BC::Loadout` free-function helpers. `lastUsedFtmwPresetName` is the
/// reserved name used for the per-loadout `__LastUsed__` sentinel preset.
namespace BC::Store::LM {
/// \brief Top-level QSettings group that holds all loadout state.
inline constexpr QLatin1StringView key{"Loadouts"};
/// \brief Field that stores the active loadout name.
inline constexpr QLatin1StringView current{"currentLoadout"};
/// \brief Field that stores the default loadout name.
inline constexpr QLatin1StringView defaultName{"defaultLoadout"};
/// \brief Array group that lists all known loadout names.
inline constexpr QLatin1StringView namesKey{"names"};
/// \brief Field used inside name-array entries to hold the loadout or preset name.
inline constexpr QLatin1StringView nameField{"name"};
/// \brief Array group that holds the loadout's hardware map entries.
inline constexpr QLatin1StringView hwMapKey{"hardwareMap"};
/// \brief Field that records which digitizer hardware key a preset was captured against.
inline constexpr QLatin1StringView digiHwKeyField{"DigiHwKey"};
/// \brief Sub-group that holds an FTMW preset's RF scalar fields.
inline constexpr QLatin1StringView rfScalarsKey{"rfScalars"};
/// \brief Array group that holds an FTMW preset's RF clock entries.
inline constexpr QLatin1StringView rfClocksKey{"rfClocks"};
/// \brief Sub-group that holds an FTMW preset's chirp scalar fields.
inline constexpr QLatin1StringView chirpScalarsKey{"chirpScalars"};
/// \brief Array group that holds an FTMW preset's chirp segment table.
inline constexpr QLatin1StringView chirpSegmentsKey{"chirpSegments"};
/// \brief Array group that holds an FTMW preset's chirp marker table.
inline constexpr QLatin1StringView chirpMarkersKey{"chirpMarkers"};
/// \brief Sub-group that holds an FTMW preset's digitizer scalar fields.
inline constexpr QLatin1StringView digiScalarsKey{"digiScalars"};
/// \brief Array group that holds an FTMW preset's digitizer analog channel entries.
inline constexpr QLatin1StringView digiAnalogKey{"digiAnalog"};
/// \brief Array group that holds an FTMW preset's digitizer digital channel entries.
inline constexpr QLatin1StringView digiDigitalKey{"digiDigital"};
/// \brief Sub-group under each loadout that holds its FTMW presets.
inline constexpr QLatin1StringView ftmwPresetsKey{"ftmwPresets"};
/// \brief Array group that lists the FTMW preset names owned by a loadout.
inline constexpr QLatin1StringView ftmwPresetNamesKey{"ftmwPresetNames"};
/// \brief Field that names a loadout's currently active FTMW preset.
inline constexpr QLatin1StringView currentFtmwPresetKey{"currentFtmwPreset"};
/// \brief Field that records the last-modified timestamp of a loadout or preset.
inline constexpr QLatin1StringView lastModifiedKey{"lastModified"};
/// \brief Reserved preset name used for the per-loadout last-used sentinel.
inline constexpr QLatin1StringView lastUsedFtmwPresetName{"__LastUsed__"};
}

class LoadoutManagerTest;

/// \brief Singleton that owns the persistent collection of `HardwareLoadout` records and their FTMW presets.
///
/// `LoadoutManager` is the sole writer of the `Loadouts/` QSettings
/// subtree. It loads every known loadout into an in-memory cache at
/// construction, services CRUD operations against that cache, and
/// flushes changes back to QSettings under the appropriate sub-groups.
/// Read access is unrestricted; write access is funneled through the
/// public CRUD methods, which emit Qt signals so the UI can refresh in
/// response to changes from any caller.
///
/// All public methods are thread-safe via an internal `QMutex`. The
/// helper signals (`loadoutAdded`, `currentFtmwPresetChanged`, etc.) are
/// emitted on the thread of the caller that triggered the change.
class LoadoutManager : public QObject, public SettingsStorage
{
    Q_OBJECT

public:
    /// \brief Access the process-wide singleton.
    static LoadoutManager &instance();
    ~LoadoutManager() override;

    // Loadout CRUD

    /// \brief Names of all loadouts currently known to the manager.
    QStringList loadoutNames() const;
    /// \brief Whether a loadout with the given name exists in the cache.
    bool loadoutExists(const QString &name) const;
    /// \brief Fetch a copy of the named loadout, or `std::nullopt` if it does not exist.
    std::optional<HardwareLoadout> getLoadout(const QString &name) const;
    /// \brief Insert or replace a loadout, persisting it to QSettings and emitting the appropriate signal.
    bool putLoadout(const HardwareLoadout &loadout);
    /// \brief Remove the named loadout, including all of its FTMW presets, from cache and QSettings.
    bool removeLoadout(const QString &name);

    // Current/default loadout

    /// \brief Name of the loadout marked active for the running session.
    QString currentLoadoutName() const;
    /// \brief Set the active loadout, emitting `currentLoadoutChanged`.
    void setCurrentLoadoutName(const QString &name);
    /// \brief Convenience accessor that returns the active loadout, if any.
    std::optional<HardwareLoadout> currentLoadout() const;

    /// \brief Name of the loadout selected on application startup when no other selection is in force.
    QString defaultLoadoutName() const;
    /// \brief Set the default loadout, emitting `defaultLoadoutChanged`.
    void setDefaultLoadoutName(const QString &name);
    /// \brief Convenience accessor that returns the default loadout, if any.
    std::optional<HardwareLoadout> defaultLoadout() const;

    /// \brief Names of all loadouts whose member set includes the given profile identity.
    QStringList loadoutsMatchingHwKey(const QString &hwKey) const;

    // FTMW preset CRUD

    /// \brief Fetch a copy of a named preset from the named loadout.
    std::optional<FtmwPreset> getFtmwPreset(const QString &loadoutName, const QString &presetName) const;
    /// \brief Insert or replace an FTMW preset in the named loadout.
    bool putFtmwPreset(const QString &loadoutName, const QString &presetName, const FtmwPreset &preset);
    /// \brief Remove the named preset from the named loadout. Cannot remove the active preset.
    bool removeFtmwPreset(const QString &loadoutName, const QString &presetName);
    /// \brief Rename a preset within the named loadout, updating the current-preset pointer if needed.
    bool renameFtmwPreset(const QString &loadoutName, const QString &oldName, const QString &newName);
    /// \brief Whether the named loadout contains a preset with the given name.
    bool ftmwPresetExists(const QString &loadoutName, const QString &presetName) const;
    /// \brief Names of the FTMW presets owned by the named loadout.
    /// \param includeLastUsed If true, the `__LastUsed__` sentinel is included in the returned list.
    QStringList ftmwPresetNames(const QString &loadoutName, bool includeLastUsed = false) const;
    /// \brief Remove every FTMW preset from the named loadout.
    bool clearFtmwPresets(const QString &loadoutName);

    // Current/default FTMW preset

    /// \brief Name of the named loadout's currently active FTMW preset.
    QString currentFtmwPresetName(const QString &loadoutName) const;
    /// \brief Set the active FTMW preset for the named loadout, emitting `currentFtmwPresetChanged`.
    bool setCurrentFtmwPresetName(const QString &loadoutName, const QString &presetName);
    /// \brief Convenience accessor that returns the named loadout's active preset, if any.
    std::optional<FtmwPreset> currentFtmwPreset(const QString &loadoutName) const;

signals:
    /// \brief Emitted after a new loadout is inserted into the cache.
    void loadoutAdded(QString name);
    /// \brief Emitted after a loadout is removed from the cache.
    void loadoutRemoved(QString name);
    /// \brief Emitted after the contents of an existing loadout are replaced.
    void loadoutChanged(QString name);
    /// \brief Emitted after the active loadout selection changes.
    void currentLoadoutChanged(QString name);
    /// \brief Emitted after the default loadout selection changes.
    void defaultLoadoutChanged(QString name);

    /// \brief Emitted after a new FTMW preset is added to a loadout.
    void ftmwPresetAdded(QString loadoutName, QString presetName);
    /// \brief Emitted after an FTMW preset is removed from a loadout.
    void ftmwPresetRemoved(QString loadoutName, QString presetName);
    /// \brief Emitted after an FTMW preset's contents are replaced.
    void ftmwPresetChanged(QString loadoutName, QString presetName);
    /// \brief Emitted after a loadout's active FTMW preset selection changes.
    void currentFtmwPresetChanged(QString loadoutName, QString presetName);

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
