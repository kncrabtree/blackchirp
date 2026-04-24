#include <data/loadout/ftmwdigitizerloadout.h>
#include <data/experiment/digitizerconfig.h>

using namespace Qt::StringLiterals;

namespace BC::Loadout {

SettingsStorage::SettingsMap digitizerScalarsMap(const FtmwDigitizerConfig &cfg)
{
    using namespace BC::Store::Digi;
    SettingsStorage::SettingsMap map;

    map[trigCh.toString()] = cfg.d_triggerChannel;
    map[trigSlope.toString()] = cfg.d_triggerSlope;
    map[trigDelay.toString()] = cfg.d_triggerDelayUSec;
    map[trigLevel.toString()] = cfg.d_triggerLevel;
    map[sRate.toString()] = cfg.d_sampleRate;
    map[recLen.toString()] = cfg.d_recordLength;
    map[bpp.toString()] = cfg.d_bytesPerPoint;
    map[bo.toString()] = cfg.d_byteOrder;
    map[blockAvg.toString()] = cfg.d_blockAverage;
    map[numAvg.toString()] = cfg.d_numAverages;
    map[multiRec.toString()] = cfg.d_multiRecord;
    map[multiRecNum.toString()] = cfg.d_numRecords;
    map[fidCh.toString()] = cfg.d_fidChannel;

    return map;
}

std::vector<SettingsStorage::SettingsMap> digitizerAnalogArray(const FtmwDigitizerConfig &cfg)
{
    using namespace BC::Store::Digi;
    std::vector<SettingsStorage::SettingsMap> array;

    for (const auto &[index, channel] : cfg.d_analogChannels) {
        SettingsStorage::SettingsMap map;
        map[chIndex.toString()] = index;
        map[en.toString()] = channel.enabled;
        map[fs.toString()] = channel.fullScale;
        map[offset.toString()] = channel.offset;
        array.push_back(map);
    }

    return array;
}

std::vector<SettingsStorage::SettingsMap> digitizerDigitalArray(const FtmwDigitizerConfig &cfg)
{
    using namespace BC::Store::Digi;
    std::vector<SettingsStorage::SettingsMap> array;

    for (const auto &[index, channel] : cfg.d_digitalChannels) {
        SettingsStorage::SettingsMap map;
        map[chIndex.toString()] = index;
        map[en.toString()] = channel.enabled;
        map[digInp.toString()] = channel.input;
        map[digRole.toString()] = channel.role;
        array.push_back(map);
    }

    return array;
}

FtmwDigitizerConfig ftmwDigitizerFromMaps(const QString &hwKey,
                                          const SettingsStorage::SettingsMap &scalars,
                                          const std::vector<SettingsStorage::SettingsMap> &analog,
                                          const std::vector<SettingsStorage::SettingsMap> &digital)
{
    using namespace BC::Store::Digi;
    FtmwDigitizerConfig cfg(hwKey);

    auto getTriggerSlope = [](int val) {
        return (val == DigitizerConfig::FallingEdge) ? DigitizerConfig::FallingEdge : DigitizerConfig::RisingEdge;
    };

    auto getByteOrder = [](int val) {
        return (val == DigitizerConfig::BigEndian) ? DigitizerConfig::BigEndian : DigitizerConfig::LittleEndian;
    };

    if (scalars.count(trigCh)) {
        cfg.d_triggerChannel = scalars.at(trigCh).value<int>();
    }
    if (scalars.count(trigSlope)) {
        cfg.d_triggerSlope = getTriggerSlope(scalars.at(trigSlope).value<int>());
    }
    if (scalars.count(trigDelay)) {
        cfg.d_triggerDelayUSec = scalars.at(trigDelay).value<double>();
    }
    if (scalars.count(trigLevel)) {
        cfg.d_triggerLevel = scalars.at(trigLevel).value<double>();
    }
    if (scalars.count(sRate)) {
        cfg.d_sampleRate = scalars.at(sRate).value<double>();
    }
    if (scalars.count(recLen)) {
        cfg.d_recordLength = scalars.at(recLen).value<int>();
    }
    if (scalars.count(bpp)) {
        cfg.d_bytesPerPoint = scalars.at(bpp).value<int>();
    }
    if (scalars.count(bo)) {
        cfg.d_byteOrder = getByteOrder(scalars.at(bo).value<int>());
    }
    if (scalars.count(blockAvg)) {
        cfg.d_blockAverage = scalars.at(blockAvg).value<bool>();
    }
    if (scalars.count(numAvg)) {
        cfg.d_numAverages = scalars.at(numAvg).value<int>();
    }
    if (scalars.count(multiRec)) {
        cfg.d_multiRecord = scalars.at(multiRec).value<bool>();
    }
    if (scalars.count(multiRecNum)) {
        cfg.d_numRecords = scalars.at(multiRecNum).value<int>();
    }
    if (scalars.count(fidCh)) {
        cfg.d_fidChannel = scalars.at(fidCh).value<int>();
    }

    for (const auto &channelMap : analog) {
        if (channelMap.count(chIndex)) {
            int index = channelMap.at(chIndex).value<int>();
            DigitizerConfig::AnalogChannel ch;
            if (channelMap.count(en)) {
                ch.enabled = channelMap.at(en).value<bool>();
            }
            if (channelMap.count(fs)) {
                ch.fullScale = channelMap.at(fs).value<double>();
            }
            if (channelMap.count(offset)) {
                ch.offset = channelMap.at(offset).value<double>();
            }
            cfg.d_analogChannels[index] = ch;
        }
    }

    for (const auto &channelMap : digital) {
        if (channelMap.count(chIndex)) {
            int index = channelMap.at(chIndex).value<int>();
            DigitizerConfig::DigitalChannel ch;
            if (channelMap.count(en)) {
                ch.enabled = channelMap.at(en).value<bool>();
            }
            if (channelMap.count(digInp)) {
                ch.input = channelMap.at(digInp).value<bool>();
            }
            if (channelMap.count(digRole)) {
                ch.role = channelMap.at(digRole).value<int>();
            }
            cfg.d_digitalChannels[index] = ch;
        }
    }

    return cfg;
}

} // namespace BC::Loadout
