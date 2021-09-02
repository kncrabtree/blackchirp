#ifndef TEMPERATURECONTROLLERCONFIG_H
#define TEMPERATURECONTROLLERCONFIG_H

#include <data/storage/headerstorage.h>

#include <QVector>

namespace BC::Store::TempControlConfig {
static const QString key{"TemperatureController"};
static const QString channel{"Channel"};
static const QString name{"Name"};
static const QString enabled{"Enabled"};
}

class TemperatureControllerConfig : public HeaderStorage
{
public:
    struct TcChannel {
        double t{0.0};
        QString name{""};
        bool enabled{false};
    };

    TemperatureControllerConfig();

    void setNumChannels(int n);

    void setTemperature(int ch, double t);
    void setName(int ch, QString n);
    void setEnabled(int ch, bool en);

    int numChannels() const { return d_channels.size(); }
    double temperature(int ch) const;
    QString channelName(int ch) const;
    bool channelEnabled(int ch) const;


private:
    QVector<TcChannel> d_channels;

    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
};

#endif // TEMPERATURECONTROLLERCONFIG_H
