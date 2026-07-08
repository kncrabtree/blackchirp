#include <data/loadout/hardwareloadout.h>

#include <optional>

namespace BC::Loadout {

using namespace BC::Store::RFC;

Map rfConfigScalarsMap(const RfConfigSnapshot &snap)
{
    Map map;
    map[commonLO] = snap.commonUpDownLO;
    map[awgM]     = snap.awgMult;
    map[upSB]     = static_cast<int>(snap.upMixSideband);
    map[chirpM]   = snap.chirpMult;
    map[downSB]   = static_cast<int>(snap.downMixSideband);
    return map;
}

Maps rfConfigClocksArray(const RfConfigSnapshot &snap)
{
    Maps array;
    for (auto it = snap.clocks.constBegin(); it != snap.clocks.constEnd(); ++it) {
        const auto &cf = it.value();
        Map map;
        map[clockType]    = static_cast<int>(it.key());
        map[hwKey]        = cf.hwKey;
        map[clockOutput]  = cf.output;
        map[clockOp]      = static_cast<int>(cf.op);
        map[clockFactor]  = cf.factor;
        map[clockFreqMHz] = cf.desiredFreqMHz;
        array.push_back(std::move(map));
    }
    return array;
}

RfConfigSnapshot rfConfigSnapshotFromMaps(const Map &scalars, const Maps &clocks)
{
    RfConfigSnapshot snap;

    if (scalars.contains(commonLO)) snap.commonUpDownLO = scalars.at(commonLO).value<bool>();
    if (scalars.contains(awgM))     snap.awgMult        = scalars.at(awgM).value<double>();
    if (scalars.contains(upSB))     snap.upMixSideband  = static_cast<RfConfig::Sideband>(scalars.at(upSB).value<int>());
    if (scalars.contains(chirpM))   snap.chirpMult      = scalars.at(chirpM).value<double>();
    if (scalars.contains(downSB))   snap.downMixSideband = static_cast<RfConfig::Sideband>(scalars.at(downSB).value<int>());

    for (const auto &m : clocks) {
        if (!m.contains(clockType) || !m.contains(hwKey))
            continue;
        RfConfig::ClockType ct = static_cast<RfConfig::ClockType>(m.at(clockType).value<int>());
        RfConfig::ClockFreq cf;
        if (m.contains(hwKey))   cf.hwKey           = m.at(hwKey).value<QString>();
        if (m.contains(clockOutput))  cf.output          = m.at(clockOutput).value<int>();
        if (m.contains(clockOp))      cf.op              = static_cast<RfConfig::MultOperation>(m.at(clockOp).value<int>());
        if (m.contains(clockFactor))  cf.factor          = m.at(clockFactor).value<double>();
        if (m.contains(clockFreqMHz)) cf.desiredFreqMHz  = m.at(clockFreqMHz).value<double>();
        snap.clocks.insert(ct, cf);
    }

    return snap;
}

void copyClocksMatching(const RfConfigSnapshot &source,
                        RfConfigSnapshot &dest,
                        const std::set<QString> &allowedHwKeys)
{
    for (auto it = source.clocks.constBegin(); it != source.clocks.constEnd(); ++it) {
        if (allowedHwKeys.count(it.value().hwKey))
            dest.clocks.insert(it.key(), it.value());
    }
}

void copyRfScalars(const RfConfigSnapshot &source, RfConfigSnapshot &dest)
{
    dest.commonUpDownLO  = source.commonUpDownLO;
    dest.awgMult         = source.awgMult;
    dest.upMixSideband   = source.upMixSideband;
    dest.chirpMult       = source.chirpMult;
    dest.downMixSideband = source.downMixSideband;
}

Map lifConversionScalarsMap(const LifConversionSnapshot &snap)
{
    using namespace BC::Store::LIFC;
    Map map;
    map[laserKey] = snap.laserKey;
    return map;
}

Maps lifConversionWiringArray(const LifConversionSnapshot &snap)
{
    using namespace BC::Store::LIFC;
    Maps array;
    array.reserve(snap.wiring.size());
    for(const auto &w : snap.wiring)
    {
        Map map;
        map[stageKey] = w.stageKey;
        map[isFinal]  = w.isFinal;

        if(w.inputs.size() > 0)
        {
            const auto &in0 = w.inputs[0];
            map[in0Type]  = static_cast<int>(in0.type);
            map[in0Key]   = in0.stageKey;
            map[in0Fixed] = in0.fixedCm1;
        }
        if(w.inputs.size() > 1)
        {
            const auto &in1 = w.inputs[1];
            map[in1Type]  = static_cast<int>(in1.type);
            map[in1Key]   = in1.stageKey;
            map[in1Fixed] = in1.fixedCm1;
        }

        array.push_back(std::move(map));
    }
    return array;
}

LifConversionSnapshot lifConversionSnapshotFromMaps(const Map &scalars, const Maps &wiring)
{
    using namespace BC::Store::LIFC;

    LifConversionSnapshot snap;
    if(scalars.contains(laserKey))
        snap.laserKey = scalars.at(laserKey).value<QString>();

    // One ordered input slot: present iff the row recorded a RefType for it.
    auto readInput = [](const Map &m, QLatin1StringView typeKey, QLatin1StringView refKey,
                        QLatin1StringView fixedKey) -> std::optional<BC::LifConv::InputRef>
    {
        if(!m.contains(typeKey))
            return std::nullopt;

        BC::LifConv::InputRef ref;
        ref.type = static_cast<BC::LifConv::RefType>(m.at(typeKey).value<int>());
        if(m.contains(refKey))
            ref.stageKey = m.at(refKey).value<QString>();
        if(m.contains(fixedKey))
            ref.fixedCm1 = m.at(fixedKey).value<double>();
        return ref;
    };

    snap.wiring.reserve(wiring.size());
    for(const auto &m : wiring)
    {
        if(!m.contains(stageKey))
            continue;

        BC::LifConv::StageWiring w;
        w.stageKey = m.at(stageKey).value<QString>();
        if(m.contains(isFinal))
            w.isFinal = m.at(isFinal).value<bool>();

        if(auto in0 = readInput(m, in0Type, in0Key, in0Fixed))
            w.inputs.push_back(*in0);
        if(auto in1 = readInput(m, in1Type, in1Key, in1Fixed))
            w.inputs.push_back(*in1);

        snap.wiring.push_back(std::move(w));
    }

    return snap;
}

Maps hardwareMapArray(const std::map<QString, QString, std::less<>> &hwMap)
{
    Maps array;
    array.reserve(hwMap.size());
    for (const auto &[k, v] : hwMap) {
        Map map;
        map[hwKey]  = k;
        map[hwImpl] = v;
        array.push_back(std::move(map));
    }
    return array;
}

std::map<QString, QString, std::less<>> hardwareMapFromArray(const Maps &array)
{
    std::map<QString, QString, std::less<>> result;
    for (const auto &m : array) {
        if (m.contains(hwKey) && m.contains(hwImpl))
            result[m.at(hwKey).value<QString>()] = m.at(hwImpl).value<QString>();
    }
    return result;
}

} // namespace BC::Loadout
