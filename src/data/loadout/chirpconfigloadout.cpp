#include <data/loadout/chirpconfigloadout.h>

namespace BC::Loadout {

using namespace BC::Store::CC;
using Map = SettingsStorage::SettingsMap;

SettingsStorage::SettingsMap chirpConfigScalarsMap(const ChirpConfig &cc)
{
    Map map;
    map[numChirps]   = cc.numChirps();
    map[interval]    = cc.chirpInterval();
    map[allIdentical] = cc.allChirpsIdentical();
    return map;
}

std::vector<SettingsStorage::SettingsMap> chirpConfigSegmentsArray(const ChirpConfig &cc)
{
    std::vector<Map> array;
    const auto &cl = cc.chirpList();
    for (int ci = 0; ci < cl.size(); ++ci) {
        const auto &segs = cl.at(ci);
        for (int si = 0; si < segs.size(); ++si) {
            const auto &seg = segs.at(si);
            Map map;
            map[chirpIndex]   = ci;
            map[segmentIndex] = si;
            map[startFreqMHz] = seg.startFreqMHz;
            map[endFreqMHz]   = seg.endFreqMHz;
            map[durationUs]   = seg.durationUs;
            map[alphaUs]      = seg.alphaUs;
            map[segEmpty]     = seg.empty;
            array.push_back(std::move(map));
        }
    }
    return array;
}

std::vector<SettingsStorage::SettingsMap> chirpConfigMarkersArray(const ChirpConfig &cc)
{
    std::vector<Map> array;
    for (const auto &m : cc.markerChannels()) {
        Map map;
        map[markerName]    = m.name;
        map[markerRole]    = static_cast<int>(m.role);
        map[timingMode]    = static_cast<int>(m.timingMode);
        map[markerStart]   = m.startTime;
        map[markerEnd]     = m.endTime;
        map[markerEnabled] = m.enabled;
        array.push_back(std::move(map));
    }
    return array;
}

ChirpConfig chirpConfigFromMaps(const SettingsStorage::SettingsMap &scalars,
                                const std::vector<SettingsStorage::SettingsMap> &segments,
                                const std::vector<SettingsStorage::SettingsMap> &markers,
                                double awgSampleRateSps)
{
    ChirpConfig cc;
    cc.setAwgSampleRate(awgSampleRateSps);

    int nChirps = 0;
    if (scalars.contains(numChirps))
        nChirps = scalars.at(numChirps).value<int>();
    cc.setNumChirps(nChirps);

    if (scalars.contains(interval))
        cc.setChirpInterval(scalars.at(interval).value<double>());

    QVector<QVector<ChirpConfig::ChirpSegment>> cl(nChirps);
    for (const auto &m : segments) {
        if (!m.contains(chirpIndex) || !m.contains(segmentIndex))
            continue;
        int ci = m.at(chirpIndex).value<int>();
        int si = m.at(segmentIndex).value<int>();
        if (ci < 0 || ci >= cl.size())
            continue;
        ChirpConfig::ChirpSegment seg;
        if (m.contains(startFreqMHz)) seg.startFreqMHz = m.at(startFreqMHz).value<double>();
        if (m.contains(endFreqMHz))   seg.endFreqMHz   = m.at(endFreqMHz).value<double>();
        if (m.contains(durationUs))   seg.durationUs   = m.at(durationUs).value<double>();
        if (m.contains(alphaUs))      seg.alphaUs      = m.at(alphaUs).value<double>();
        if (m.contains(segEmpty))     seg.empty        = m.at(segEmpty).value<bool>();
        if (si == cl[ci].size())
            cl[ci].push_back(seg);
        else if (si >= 0 && si < cl[ci].size())
            cl[ci][si] = seg;
    }
    cc.setChirpList(cl);

    QVector<MarkerChannel> mchans;
    for (const auto &m : markers) {
        MarkerChannel mc;
        if (m.contains(markerName))    mc.name       = m.at(markerName).value<QString>();
        if (m.contains(markerRole))    mc.role        = static_cast<MarkerRole>(m.at(markerRole).value<int>());
        if (m.contains(timingMode))    mc.timingMode  = static_cast<MarkerChannel::TimingMode>(m.at(timingMode).value<int>());
        if (m.contains(markerStart))   mc.startTime   = m.at(markerStart).value<double>();
        if (m.contains(markerEnd))     mc.endTime     = m.at(markerEnd).value<double>();
        if (m.contains(markerEnabled)) mc.enabled     = m.at(markerEnabled).value<bool>();
        mchans.push_back(mc);
    }
    cc.setMarkerChannels(mchans);

    return cc;
}

} // namespace BC::Loadout
