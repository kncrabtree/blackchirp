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
        Setpoint,
        Name
    };

    struct ChannelConfig {
        bool enabled;
        double setpoint;
        QString name;
    };

    QVariant setting(int index, FlowConfig::Setting s) const;
    double pressureSetPoint() const;
    int size() const;

    void add(double set = 0.0, QString name = QString(""));
    void set(int index, FlowConfig::Setting s, QVariant val);
    void setPressure(double p);
    void setPressureSetpoint(double s);

private:
    QSharedDataPointer<FlowConfigData> data;
};


class FlowConfigData : public QSharedData
{
public:
    FlowConfigData() : pressureSetpoint(0.0) {}

    QList<FlowConfig::ChannelConfig> flowList;
    double pressureSetpoint;
};

Q_DECLARE_METATYPE(FlowConfig::Setting)

#endif // FLOWCONFIG_H
