#ifndef FLOWCONFIG_H
#define FLOWCONFIG_H

#include <QSharedDataPointer>

#include <QList>
#include <QVariant>
#include <QMetaType>

#include "datastructs.h"

class FlowConfigData;

class FlowConfig
{
public:
    FlowConfig();
    FlowConfig(const FlowConfig &);
    FlowConfig &operator=(const FlowConfig &);
    ~FlowConfig();

    QVariant setting(int index, BlackChirp::FlowSetting s) const;
    double pressureSetpoint() const;
    double pressure() const;
    bool pressureControlMode() const;
    int size() const;

    void add(double set = 0.0, QString name = QString(""));
    void set(int index, BlackChirp::FlowSetting s, QVariant val);
    void setPressure(double p);
    void setPressureSetpoint(double s);
    void setPressureControlMode(bool en);

    QMap<QString, QPair<QVariant,QString>> headerMap() const;
    void parseLine(const QString key, const QVariant val);

private:
    QSharedDataPointer<FlowConfigData> data;
};


class FlowConfigData : public QSharedData
{
public:
    FlowConfigData() : pressureSetpoint(0.0), pressure(0.0), pressureControlMode(false) {}

    QList<BlackChirp::FlowChannelConfig> configList;
    double pressureSetpoint;

    QList<double> flowList;
    double pressure;
    bool pressureControlMode;
};

#endif // FLOWCONFIG_H
