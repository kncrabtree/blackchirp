#ifndef FLOWCONFIG_H
#define FLOWCONFIG_H

#include <QSharedDataPointer>

#include <QList>
#include <QVariant>
#include <QMetaType>

#define BC_FLOW_NUMCHANNELS 4

class FlowConfigData;

class FlowConfig
{
public:
    FlowConfig();
    FlowConfig(const FlowConfig &);
    FlowConfig &operator=(const FlowConfig &);
    ~FlowConfig();

    enum Setting {
        Setpoint,
        Flow,
        Name
    };

    struct ChannelConfig {
        bool enabled;
        double setpoint;
        double flow;
        QString name;
    };

    QVariant setting(int index, FlowConfig::Setting s) const;
    double pressure() const;
    double pressureSetPoint() const;

    void set(int index, FlowConfig::Setting s, QVariant val);
    void setPressure(double p);
    void setPressureSetpoint(double s);

private:
    QSharedDataPointer<FlowConfigData> data;
};


class FlowConfigData : public QSharedData
{
public:
    FlowConfigData() : pressureSetpoint(0.0), pressure(0.0) {}

    QList<FlowConfig::ChannelConfig> flowList;
    double pressureSetpoint;
    double pressure;
};

Q_DECLARE_METATYPE(FlowConfig::Setting)

#endif // FLOWCONFIG_H
