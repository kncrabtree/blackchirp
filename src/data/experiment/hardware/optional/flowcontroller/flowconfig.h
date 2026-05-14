#ifndef FLOWCONFIG_H
#define FLOWCONFIG_H

#include <data/storage/headerstorage.h>

#include <QVector>
#include <QVariant>
#include <QMetaType>


namespace BC::Store::FlowConfig {
inline constexpr QLatin1StringView channel{"Channel"};
inline constexpr QLatin1StringView name{"Name"};
inline constexpr QLatin1StringView enabled{"Enabled"};
inline constexpr QLatin1StringView setPoint{"FlowSetPoint"};
inline constexpr QLatin1StringView pSetPoint{"PressureSetPoint"};
inline constexpr QLatin1StringView pcEnabled{"PressureControlEnabled"};
}

class FlowConfig : public HeaderStorage
{
    Q_GADGET
public:
    struct FlowChannel {
        bool enabled{false};
        double setpoint{0.0};
        double flow{0.0};
        QString name{""};
    };


    enum FlowChSetting {
        Enabled,
        Setpoint,
        Flow,
        Name
    };
    Q_ENUM(FlowChSetting)

    FlowConfig(const QString& hwKey);
    ~FlowConfig();

    double d_pressureSetpoint{0.0};
    double d_pressure{0.0};
    bool d_pressureControlMode{false};

    QVariant setting(int index, FlowChSetting s) const;
    int size() const;

    void addCh(double set = 0.0, QString name = QString(""));
    void addCh(double set, QString name, bool enabled);
    void setCh(int index, FlowChSetting s, QVariant val);

private:
    QList<FlowConfig::FlowChannel> d_configList;

    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
};

Q_DECLARE_METATYPE(FlowConfig)
Q_DECLARE_METATYPE(FlowConfig::FlowChSetting)
Q_DECLARE_TYPEINFO(FlowConfig::FlowChannel,Q_MOVABLE_TYPE);

#endif // FLOWCONFIG_H
