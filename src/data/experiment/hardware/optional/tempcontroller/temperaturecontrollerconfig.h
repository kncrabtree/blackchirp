#ifndef TEMPERATURECONTROLLERCONFIG_H
#define TEMPERATURECONTROLLERCONFIG_H

#include <data/storage/headerstorage.h>

namespace BC::Store::TempControlConfig {
inline constexpr QLatin1StringView channel{"Channel"};
inline constexpr QLatin1StringView name{"Name"};
inline constexpr QLatin1StringView enabled{"Enabled"};
}

class TemperatureControllerConfig : public HeaderStorage
{
public:
    struct TcChannel {
        double t{0.0};
        QString name{""};
        bool enabled{false};
    };

    TemperatureControllerConfig(const QString& hwKey);

    void setNumChannels(int n);

    void setTemperature(uint ch, double t);
    void setName(uint ch, QString n);
    void setEnabled(uint ch, bool en);

    uint numChannels() const { return d_channels.size(); }
    double temperature(uint ch) const;
    QString channelName(uint ch) const;
    bool channelEnabled(uint ch) const;


private:
    std::vector<TcChannel> d_channels;

    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
};

#endif // TEMPERATURECONTROLLERCONFIG_H
