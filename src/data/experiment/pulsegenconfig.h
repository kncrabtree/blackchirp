#ifndef PULSEGENCONFIG_H
#define PULSEGENCONFIG_H

#include <data/storage/headerstorage.h>

#include <QVector>
#include <QVariant>
#include <QMap>

namespace BC::Key::PGenConfig {
static const QString key{"PulseGenerator"};
}

namespace BC::Store::PGenConfig {
static const QString rate("RepRate");
static const QString channel("Channel");
static const QString delay("Delay");
static const QString width("Width");
static const QString level("ActiveLevel");
static const QString role("Role");
static const QString name("Name");
static const QString enabled("Enabled");
}

class PulseGenConfig : public HeaderStorage
{
    Q_GADGET
public:
    enum ActiveLevel { ActiveInvalid = -1, ActiveLow, ActiveHigh };
    enum Setting { DelaySetting, WidthSetting, EnabledSetting, LevelSetting, NameSetting, RoleSetting };
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
        ,LIF
#endif
#ifdef BC_MOTOR
        ,Motor
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
        Role role{None};
    };

    PulseGenConfig();
    ~PulseGenConfig();

    ChannelConfig at(const int i) const;
    int size() const;
    bool isEmpty() const;
    QVariant setting(const int index, const Setting s) const;
    QVector<QVariant> setting(Role role, const Setting s) const;
    ChannelConfig settings(const int index) const;
    QVector<int> channelsForRole(Role role) const;
    double repRate() const;

    void set(const int index, const Setting s, const QVariant val);
    void set(const int index, const ChannelConfig cc);
    void set(Role role, const Setting s, const QVariant val);
    void set(Role role, const ChannelConfig cc);
    void addChannel();
    void setRepRate(const double r);

private:
    QVector<PulseGenConfig::ChannelConfig> d_config;
    double d_repRate{1.0};

    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
};

Q_DECLARE_METATYPE(PulseGenConfig)
Q_DECLARE_TYPEINFO(PulseGenConfig::ChannelConfig,Q_MOVABLE_TYPE);

#endif // PULSEGENCONFIG_H
