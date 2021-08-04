#ifndef FLOWCONFIG_H
#define FLOWCONFIG_H

#include <QSharedDataPointer>

#include <QList>
#include <QVariant>
#include <QMetaType>

#include <data/datastructs.h>

class FlowConfigData;

class FlowConfig
{
    Q_GADGET
public:
    struct FlowChannel {
        bool enabled;
        double setpoint;
        double flow;
        QString name;
    };


    enum FlowChSetting {
        Enabled,
        Setpoint,
        Flow,
        Name
    };
    Q_ENUM(FlowChSetting)

    FlowConfig();
    FlowConfig(const FlowConfig &);
    FlowConfig &operator=(const FlowConfig &);
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
    QSharedDataPointer<FlowConfigData> data;
};


class FlowConfigData : public QSharedData
{
public:
    FlowConfigData() : pressureSetpoint(0.0), pressure(0.0), pressureControlMode(false) {}

    QList<FlowConfig::FlowChannel> configList;
    double pressureSetpoint;

    double pressure;
    bool pressureControlMode;
};

Q_DECLARE_TYPEINFO(FlowConfig::FlowChannel,Q_MOVABLE_TYPE);

#endif // FLOWCONFIG_H
