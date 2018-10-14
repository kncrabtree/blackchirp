#ifndef PULSEGENCONFIG_H
#define PULSEGENCONFIG_H

#include <QSharedDataPointer>

#include <QList>
#include <QVariant>
#include <QMap>

#include "datastructs.h"


class PulseGenConfigData;

class PulseGenConfig
{
public:
    PulseGenConfig();
    PulseGenConfig(const PulseGenConfig &);
    PulseGenConfig &operator=(const PulseGenConfig &);
    ~PulseGenConfig();

    BlackChirp::PulseChannelConfig at(const int i) const;
    int size() const;
    bool isEmpty() const;
    QVariant setting(const int index, const BlackChirp::PulseSetting s) const;
    BlackChirp::PulseChannelConfig settings(const int index) const;
    double repRate() const;
    QMap<QString,QPair<QVariant,QString>> headerMap() const;
    void parseLine(QString key, QVariant val);

    void set(const int index, const BlackChirp::PulseSetting s, const QVariant val);
    void set(const int index, const BlackChirp::PulseChannelConfig cc);
    void set(BlackChirp::PulseRole role, const BlackChirp::PulseSetting s, const QVariant val);
    void set(BlackChirp::PulseRole role, const BlackChirp::PulseChannelConfig cc);
    void add(const QString name, const bool enabled, const double delay, const double width, const BlackChirp::PulseActiveLevel level, const BlackChirp::PulseRole role = BlackChirp::NoPulseRole);
    void setRepRate(const double r);

private:
    QSharedDataPointer<PulseGenConfigData> data;
};

class PulseGenConfigData : public QSharedData
{
public:
    QList<BlackChirp::PulseChannelConfig> config;
    double repRate;
};

#endif // PULSEGENCONFIG_H
