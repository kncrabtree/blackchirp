#include <data/loadout/rfconfigsnapshot.h>

RfConfigSnapshot RfConfigSnapshot::fromRfConfig(const RfConfig &c)
{
    RfConfigSnapshot snap;
    snap.commonUpDownLO = c.d_commonUpDownLO;
    snap.awgMult = c.d_awgMult;
    snap.upMixSideband = c.d_upMixSideband;
    snap.chirpMult = c.d_chirpMult;
    snap.downMixSideband = c.d_downMixSideband;
    snap.clocks = c.getClocks();
    return snap;
}

void RfConfigSnapshot::applyTo(RfConfig &c) const
{
    c.d_commonUpDownLO = commonUpDownLO;
    c.d_awgMult = awgMult;
    c.d_upMixSideband = upMixSideband;
    c.d_chirpMult = chirpMult;
    c.d_downMixSideband = downMixSideband;
    c.setCurrentClocks(clocks);
}
