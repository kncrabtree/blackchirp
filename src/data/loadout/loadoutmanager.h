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
inline constexpr QLatin1StringView ftmwPresentKey{"ftmwPresent"};
inline constexpr QLatin1StringView digiHwKeyField{"DigiHwKey"};
inline constexpr QLatin1StringView hwMapKey{"hardwareMap"};
inline constexpr QLatin1StringView rfConfigKey{"rfConfig"};
inline constexpr QLatin1StringView rfClocksKey{"rfClocks"};
inline constexpr QLatin1StringView chirpScalarsKey{"chirpScalars"};
inline constexpr QLatin1StringView chirpSegmentsKey{"chirpSegments"};
inline constexpr QLatin1StringView chirpMarkersKey{"chirpMarkers"};
inline constexpr QLatin1StringView digiScalarsKey{"digiScalars"};
inline constexpr QLatin1StringView digiAnalogKey{"digiAnalog"};
inline constexpr QLatin1StringView digiDigitalKey{"digiDigital"};
}

class LoadoutManagerTest;

class LoadoutManager : public QObject, public SettingsStorage
{
    Q_OBJECT

public:
    static LoadoutManager &instance();
    ~LoadoutManager() override;

    QStringList loadoutNames() const;
    bool loadoutExists(const QString &name) const;
    std::optional<HardwareLoadout> getLoadout(const QString &name) const;
    bool putLoadout(const HardwareLoadout &loadout);
    bool removeLoadout(const QString &name);

    QString currentLoadoutName() const;
    void setCurrentLoadoutName(const QString &name);
    std::optional<HardwareLoadout> currentLoadout() const;

    QString defaultLoadoutName() const;
    void setDefaultLoadoutName(const QString &name);
    std::optional<HardwareLoadout> defaultLoadout() const;

    QStringList loadoutsMatchingHwKey(const QString &hwKey) const;

signals:
    void loadoutAdded(QString name);
    void loadoutRemoved(QString name);
    void loadoutChanged(QString name);
    void currentLoadoutChanged(QString name);
    void defaultLoadoutChanged(QString name);

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

    QHash<QString, HardwareLoadout> d_loadouts;
    QString d_current;
    QString d_default;
    mutable QMutex d_mutex;

    static LoadoutManager *s_instance;
    friend class LoadoutManagerTest;
};

#endif // BC_LOADOUTMANAGER_H
