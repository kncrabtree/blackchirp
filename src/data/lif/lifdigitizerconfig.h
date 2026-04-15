#ifndef LIFDIGITIZERCONFIG_H
#define LIFDIGITIZERCONFIG_H

#include <data/experiment/digitizerconfig.h>

namespace BC::Store::Digi{
inline constexpr QLatin1StringView lifChannel("LifChannel");
inline constexpr QLatin1StringView lifRefChannel("LifRefChannel");
inline constexpr QLatin1StringView lifRefEnabled("LifRefEnabled");
inline constexpr QLatin1StringView lifChannelOrder("LifChannelOrder");
}

class LifDigitizerConfig : public DigitizerConfig
{
public:
    LifDigitizerConfig(const QString& hwKey);

    enum ChannelOrder {
        Sequential,
        Interleaved
    };
    Q_ENUM(ChannelOrder)

    int d_lifChannel{1};
    int d_refChannel{2};
    bool d_refEnabled{false};
    ChannelOrder d_channelOrder{Interleaved};

    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
};

#endif // LIFDIGITIZERCONFIG_H
