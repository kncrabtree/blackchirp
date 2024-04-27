#ifndef LIFDIGITIZERCONFIG_H
#define LIFDIGITIZERCONFIG_H

#include <data/experiment/digitizerconfig.h>

namespace BC::Store::Digi{
static const QString lifChannel("LifChannel");
static const QString lifRefChannel("LifRefChannel");
static const QString lifRefEnabled("LifRefEnabled");
static const QString lifChannelOrder("LifChannelOrder");
}

class LifDigitizerConfig : public DigitizerConfig
{
public:
    LifDigitizerConfig(const QString subKey);

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
