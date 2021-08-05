#ifndef FLOWCONFIG_H
#define FLOWCONFIG_H

#include <data/storage/headerstorage.h>

#include <QVector>
#include <QVariant>
#include <QMetaType>


namespace BC::Store::FlowConfig {
static const QString key("FlowController");
static const QString channel("Channel");
static const QString name("Name");
static const QString enabled("Enabled");
static const QString setPoint("FlowSetPoint");
static const QString pSetPoint("PressureSetPoint");
static const QString pcEnabled("PressureControlEnabled");
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

    FlowConfig();
    ~FlowConfig();

    QVariant setting(int index, FlowChSetting s) const;
    double pressureSetpoint() const;
    double pressure() const;
    bool pressureControlMode() const;
    int size() const;

    void add(double set = 0.0, QString name = QString(""));
    void set(int index, FlowChSetting s, QVariant val);
    void setPressure(double p);
    void setPressureSetpoint(double s);
    void setPressureControlMode(bool en);

private:
    QVector<FlowConfig::FlowChannel> d_configList;
    double d_pressureSetpoint{0.0};

    double d_pressure{0.0};
    bool d_pressureControlMode{false};

    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
};

Q_DECLARE_METATYPE(FlowConfig)
Q_DECLARE_METATYPE(FlowConfig::FlowChSetting);
Q_DECLARE_TYPEINFO(FlowConfig::FlowChannel,Q_MOVABLE_TYPE);

#endif // FLOWCONFIG_H
