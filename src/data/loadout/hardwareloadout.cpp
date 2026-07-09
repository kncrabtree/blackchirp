#include <data/loadout/hardwareloadout.h>

#include <optional>

#include <QMetaEnum>

#include <data/loghandler.h>
#include <data/storage/enumcsvconvert.h>

namespace BC::Loadout {

using namespace BC::Store::RFC;
using namespace Qt::StringLiterals;

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

        // RefType is a Q_ENUM_NS (BC::LifConv, lifunits.h); persist the enum
        // key name via the same BC::CSV helper used for BC::LifConv::Op in
        // liftopology.csv, so reordering RefType cannot silently remap an
        // already-saved preset.
        //
        // NOTE: InputRef has a second, independent on-disk encoding in
        // lifconfig.cpp (the compact "Fixed:<cm1>"/hwKey token written to
        // liftopology.csv). Any future change to InputRef's fields or
        // semantics must be mirrored in both serializers.
        if(w.inputs.size() > 0)
        {
            const auto &in0 = w.inputs[0];
            map[in0Type]  = BC::CSV::enumKeyName(QVariant::fromValue(in0.type));
            map[in0Key]   = in0.stageKey;
            map[in0Fixed] = in0.fixedCm1;
        }
        if(w.inputs.size() > 1)
        {
            const auto &in1 = w.inputs[1];
            map[in1Type]  = BC::CSV::enumKeyName(QVariant::fromValue(in1.type));
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
    // RefType is read back by enum key name (with a legacy-int fallback) via
    // the same BC::CSV helper used for BC::LifConv::Op. A stored value that
    // resolves to neither a known key name nor a known enumerator value
    // (a hand-edited settings file, or a legacy int outside the range this
    // build's RefType enumerates) falls back to Laser with a warning rather
    // than propagating an invalid enumerator into the conversion graph.
    auto readInput = [](const Map &m, QLatin1StringView typeKey, QLatin1StringView refKey,
                        QLatin1StringView fixedKey, const QString &ownerStageKey) -> std::optional<BC::LifConv::InputRef>
    {
        if(!m.contains(typeKey))
            return std::nullopt;

        const QVariant &typeVal = m.at(typeKey);

        BC::LifConv::InputRef ref;
        ref.type = BC::CSV::enumFromVariant<BC::LifConv::RefType>(typeVal, BC::LifConv::RefType::Laser);

        auto meta = QMetaEnum::fromType<BC::LifConv::RefType>();
        if(!meta.valueToKey(static_cast<int>(ref.type)))
        {
            bcWarn(u"Loadout preset wiring for stage \"%1\" has an unrecognized input reference type (\"%2\"); defaulting to Laser."_s
                       .arg(ownerStageKey, typeVal.toString()));
            ref.type = BC::LifConv::RefType::Laser;
        }

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

        if(auto in0 = readInput(m, in0Type, in0Key, in0Fixed, w.stageKey))
            w.inputs.push_back(*in0);
        if(auto in1 = readInput(m, in1Type, in1Key, in1Fixed, w.stageKey))
            w.inputs.push_back(*in1);

        snap.wiring.push_back(std::move(w));
    }

    return snap;
}

Maps hardwareMapArray(const std::map<QString, QString, std::less<>> &hwMap,
                      const std::map<QString, QString, std::less<>> &hwIdentity)
{
    Maps array;
    array.reserve(hwMap.size());
    for (const auto &[k, v] : hwMap) {
        Map map;
        map[hwKey]  = k;
        map[hwImpl] = v;
        // Co-locate the identity token with the impl it validates. Omit the
        // field entirely when no non-empty identity was captured for this
        // member, so a reader can distinguish "no identity known" (absent)
        // from any real token.
        auto it = hwIdentity.find(k);
        if (it != hwIdentity.end() && !it->second.isEmpty())
            map[BC::Store::RFC::hwIdentity] = it->second;
        array.push_back(std::move(map));
    }
    return array;
}

void hardwareMapFromArray(const Maps &array,
                          std::map<QString, QString, std::less<>> &hwMap,
                          std::map<QString, QString, std::less<>> &hwIdentity)
{
    hwMap.clear();
    hwIdentity.clear();
    for (const auto &m : array) {
        if (!m.contains(hwKey) || !m.contains(hwImpl))
            continue;
        const auto k = m.at(hwKey).value<QString>();
        hwMap[k] = m.at(hwImpl).value<QString>();
        // A record written before identity tracking has no Identity field;
        // leave that key absent from the identity map rather than synthesizing
        // a token.
        if (m.contains(BC::Store::RFC::hwIdentity)) {
            const auto id = m.at(BC::Store::RFC::hwIdentity).value<QString>();
            if (!id.isEmpty())
                hwIdentity[k] = id;
        }
    }
}

bool ftmwPresetReferencesHardware(const FtmwPreset &preset, const QString &hwKey)
{
    if (preset.digiHwKey == hwKey)
        return true;
    return preset.rfConfig.referencedHwKeys().count(hwKey) > 0;
}

bool lifPresetReferencesHardware(const LifPreset &preset, const QString &hwKey)
{
    if (preset.conversion.laserKey == hwKey)
        return true;
    for (const auto &w : preset.conversion.wiring) {
        if (w.stageKey == hwKey)
            return true;
    }
    return false;
}

} // namespace BC::Loadout
