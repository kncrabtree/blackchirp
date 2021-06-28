#ifndef DIGITIZERCONFIG_H
#define DIGITIZERCONFIG_H

#include <QObject>
#include <QDataStream>

namespace BC::Key::Digi {
static const QString numChannels("numChannels");
static const QString hasAuxChannel("hasAuxChannel");
static const QString minFullScale("minFullScale");
static const QString maxFullScale("maxFullScale");
static const QString minVOffset("minVOffset");
static const QString maxVOffset("maxVOffset");
static const QString minTrigDelay("minTrigDelayUs");
static const QString maxTrigDelay("maxTrigDelayUs");
static const QString minTrigLevel("minTrigLevel");
static const QString maxTrigLevel("maxTrigLevel");
static const QString maxRecordLength("maxRecordLength");
static const QString multiRecord("canMultiRecord");
static const QString maxRecords("maxRecords");
static const QString blockAverage("canBlockAverage");
static const QString maxAverages("maxAverages");
static const QString multiBlock("canBlockAndMultiRecord");
static const QString maxBytes("maxBytesPerPoint");
static const QString sampleRates("sampleRates");
static const QString srText("text");
static const QString srValue("val");
}

class DigitizerConfig
{
    Q_GADGET
public:
    struct Channel {
        int channelNum{0};
        bool enabled{false};
        double fullScale{0.0};
        double offset{0.0};
    };

    enum TriggerSlope {
        RisingEdge,
        FallingEdge
    };
    Q_ENUM(TriggerSlope)

    DigitizerConfig();

    //vertical channels
    QList<Channel> d_channels;

    //triggering
    int d_triggerChannel{0};
    TriggerSlope d_triggerSlope{RisingEdge};
    double d_triggerDelayUSec{0.0};
    double d_triggerLevel{0.0};

    //horizontal
    double d_sampleRate{0.0};
    int d_recordLength{0};

    //data transfer
    int d_bytesPerPoint{0};
    QDataStream::ByteOrder d_byteOrder{QDataStream::LittleEndian};

    //averaging
    bool d_blockAverage{false};
    int d_numAverages{0};

    //multiRecord/fastframe
    bool d_multiRecord{false};
    int d_numRecords{0};

    double xIncr() const;
    double yMult(int ch) const;


};

Q_DECLARE_TYPEINFO(DigitizerConfig::Channel,Q_PRIMITIVE_TYPE);

#endif // DIGITIZERCONFIG_H
