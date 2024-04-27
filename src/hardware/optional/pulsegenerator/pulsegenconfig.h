#ifndef PULSEGENCONFIG_H
#define PULSEGENCONFIG_H

#include <data/storage/headerstorage.h>

#include <QVector>
#include <QVariant>
#include <QMap>

namespace BC::Store::PGenConfig {
static const QString rate{"RepRate"};
static const QString channel{"Channel"};
static const QString delay{"Delay"};
static const QString width{"Width"};
static const QString level{"ActiveLevel"};
static const QString role{"Role"};
static const QString name{"Name"};
static const QString enabled{"Enabled"};
static const QString chMode{"Mode"};
static const QString sync{"SyncChannel"};
static const QString dutyOn{"DutyOn"};
static const QString dutyOff{"DutyOff"};
static const QString pGenMode{"PulseGenMode"};
static const QString pGenEnabled{"PulseGenEnabled"};
}

class PulseGenConfig : public HeaderStorage
{
    Q_GADGET
public:
    enum ActiveLevel { ActiveLow, ActiveHigh };
    enum Setting { DelaySetting, WidthSetting, EnabledSetting, LevelSetting, NameSetting, RoleSetting, ModeSetting, SyncSetting, DutyOnSetting, DutyOffSetting, RepRateSetting, PGenModeSetting, PGenEnabledSetting };
    enum Role {
        None,
        Gas,
        DC,
        AWG,
        Prot,
        Amp,
        Laser,
        XMer
#ifdef BC_LIF
        ,LIF = 999
#endif
    };
    enum PGenMode { Continuous, Triggered };
    enum ChannelMode { Normal, DutyCycle };
    Q_ENUM(ActiveLevel)
    Q_ENUM(Setting)
    Q_ENUM(Role)
    Q_ENUM(PGenMode)
    Q_ENUM(ChannelMode)

    struct ChannelConfig {
        QString channelName{""};
        bool enabled{false};
        double delay{0.0};
        double width{1.0};
        ActiveLevel level{ActiveHigh};
        Role role{None};
        ChannelMode mode{Normal};
        int syncCh{0};
        int dutyOn{1};
        int dutyOff{1};
    };


    QVector<PulseGenConfig::ChannelConfig> d_channels;
    double d_repRate{1.0};
    PGenMode d_mode{Continuous};
    bool d_pulseEnabled{true};


    PulseGenConfig(QString subKey=QString(""), int index=-1);
    ~PulseGenConfig();

    ChannelConfig at(const int i) const;
    int size() const;
    bool isEmpty() const;
    QVariant setting(const int index, const Setting s) const;
    QVariant setting(Role role, const Setting s) const;
    ChannelConfig settings(const int index) const;
    QVector<Role> activeRoles() const;
    QVector<int> channelsForRole(Role role) const;
    double channelStart(const int index) const;
    bool testCircularSync(const int index, int newSyncCh);

    void setCh(const int index, const Setting s, const QVariant val);
    void setCh(const int index, const ChannelConfig cc);
    void setCh(Role role, const Setting s, const QVariant val);
    void setCh(Role role, const ChannelConfig cc);
    void addChannel();


    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;

private:
    QString d_hwSubKey;
};

Q_DECLARE_METATYPE(PulseGenConfig)
Q_DECLARE_TYPEINFO(PulseGenConfig::ChannelConfig,Q_MOVABLE_TYPE);

#endif // PULSEGENCONFIG_H
