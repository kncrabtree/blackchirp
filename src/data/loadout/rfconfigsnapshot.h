#ifndef RFCONFIGSNAPSHOT_H
#define RFCONFIGSNAPSHOT_H

#include <QHash>

#include <data/experiment/rfconfig.h>

struct RfConfigSnapshot
{
    bool commonUpDownLO{false};
    double awgMult{1.0};
    RfConfig::Sideband upMixSideband{RfConfig::UpperSideband};
    double chirpMult{1.0};
    RfConfig::Sideband downMixSideband{RfConfig::UpperSideband};
    QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks;

    static RfConfigSnapshot fromRfConfig(const RfConfig &c);
    void applyTo(RfConfig &c) const;
};

#endif // RFCONFIGSNAPSHOT_H
