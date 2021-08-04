#ifndef PULSEGENCONFIG_H
#define PULSEGENCONFIG_H

#include <QSharedDataPointer>

#include <QVector>
#include <QVariant>
#include <QMap>


class PulseGenConfigData;

class PulseGenConfig
{
    Q_GADGET
public:
    enum ActiveLevel { ActiveInvalid = -1, ActiveLow, ActiveHigh };
    enum Setting { DelaySetting, WidthSetting, EnabledSetting, LevelSetting, NameSetting, RoleSetting };
    enum Role {
        NoRole,
        GasRole,
        DcRole,
        AwgRole,
        ProtRole,
        AmpRole,
        LaserRole,
        ExcimerRole
#ifdef BC_LIF
        ,LifRole
#endif
#ifdef BC_MOTOR
        ,MotorRole
#endif
    };
    Q_ENUM(ActiveLevel)
    Q_ENUM(Setting)
    Q_ENUM(Role)

    struct ChannelConfig {
        QString channelName{""};
        bool enabled{false};
        double delay{0.0};
        double width{1.0};
        ActiveLevel level{ActiveHigh};
        Role role{NoRole};
    };

    static const inline QMap<Role,QString> roles {
        {NoRole,"None"},
        {GasRole,"Gas"},
        {DcRole,"DC"},
        {AwgRole,"Awg"},
        {ProtRole,"Prot"},
        {AmpRole,"Amp"},
        {LaserRole,"Laser"},
        {ExcimerRole,"XMer"}
#ifdef BC_LIF
        ,{LifRole,"LIF"}
#endif
#ifdef BC_MOTOR
        ,{MotorRole,"Motor"}
#endif
    };
    static const inline std::map<Role,QString> stdRoles = roles.toStdMap();

    PulseGenConfig();
    PulseGenConfig(const PulseGenConfig &);
    PulseGenConfig &operator=(const PulseGenConfig &);
    ~PulseGenConfig();

    ChannelConfig at(const int i) const;
    int size() const;
    bool isEmpty() const;
    QVariant setting(const int index, const Setting s) const;
    QList<QVariant> setting(Role role, const Setting s) const;
    ChannelConfig settings(const int index) const;
    QList<int> channelsForRole(Role role) const;
    double repRate() const;

    void set(const int index, const Setting s, const QVariant val);
    void set(const int index, const ChannelConfig cc);
    void set(Role role, const Setting s, const QVariant val);
    void set(Role role, const ChannelConfig cc);
    void addChannel();
    void setRepRate(const double r);

private:
    QSharedDataPointer<PulseGenConfigData> data;
};

class PulseGenConfigData : public QSharedData
{
public:
    QVector<PulseGenConfig::ChannelConfig> config;
    double repRate{1.0};
};

Q_DECLARE_TYPEINFO(PulseGenConfig::ChannelConfig,Q_MOVABLE_TYPE);

#endif // PULSEGENCONFIG_H
