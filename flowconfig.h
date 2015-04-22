#ifndef FLOWCONFIG_H
#define FLOWCONFIG_H

#include <QSharedDataPointer>

#include <QList>
#include <QVariant>
#include <QMetaType>

class FlowConfigData;

class FlowConfig
{
public:
    FlowConfig();
    FlowConfig(const FlowConfig &);
    FlowConfig &operator=(const FlowConfig &);
    ~FlowConfig();

    enum Setting {
        Enabled,
        Setpoint,
        Flow,
        Name
    };

    struct ChannelConfig {
        bool enabled;
        double setpoint;
        QString name;
    };

    QVariant setting(int index, FlowConfig::Setting s) const;
    double pressureSetpoint() const;
    double pressure() const;
    bool pressureControlMode() const;
    int size() const;

    void add(double set = 0.0, QString name = QString(""));
    void set(int index, FlowConfig::Setting s, QVariant val);
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

    QList<FlowConfig::ChannelConfig> configList;
    double pressureSetpoint;

    QList<double> flowList;
    double pressure;
    bool pressureControlMode;
};

Q_DECLARE_METATYPE(FlowConfig::Setting)

#endif // FLOWCONFIG_H
