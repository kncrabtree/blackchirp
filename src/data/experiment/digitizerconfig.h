#ifndef DIGITIZERCONFIG_H
#define DIGITIZERCONFIG_H

#include <QObject>
#include <QDataStream>
#include <set>

#include <data/storage/headerstorage.h>

namespace BC::Key::Digi {
static const QString dwAnChannels{"channels"};
static const QString dwDigChannels{"digitalChannels"};
static const QString numAnalogChannels{"numAnalogChannels"};
static const QString hasAuxTriggerChannel{"hasAuxTriggerChannel"};
static const QString numDigitalChannels{"numDigitalChannels"};
static const QString minFullScale{"minFullScale"};
static const QString maxFullScale{"maxFullScale"};
static const QString minVOffset{"minVOffset"};
static const QString maxVOffset{"maxVOffset"};
static const QString isTriggered{"isTriggered"};
static const QString minTrigDelay{"minTrigDelayUs"};
static const QString maxTrigDelay{"maxTrigDelayUs"};
static const QString minTrigLevel{"minTrigLevel"};
static const QString maxTrigLevel{"maxTrigLevel"};
static const QString maxRecordLength{"maxRecordLength"};
static const QString canMultiRecord{"canMultiRecord"};
static const QString maxRecords{"maxRecords"};
static const QString canBlockAverage{"canBlockAverage"};
static const QString maxAverages{"maxAverages"};
static const QString multiBlock{"canBlockAndMultiRecord"};
static const QString maxBytes{"maxBytesPerPoint"};
static const QString sampleRates{"sampleRates"};
static const QString srText{"text"};
static const QString srValue{"val"};
}

namespace BC::Store::Digi {
static const QString an{"AnalogChannel"};
static const QString dig{"DigitalChannel"};
static const QString digInp{"Input"};
static const QString digRole{"Role"};
static const QString en{"Enabled"};
static const QString fs{"FullScale"};
static const QString offset{"VerticalOffset"};
static const QString trigSlope{"TriggerEdge"};
static const QString trigCh{"TriggerChannel"};
static const QString trigDelay{"TriggerDelay"};
static const QString trigLevel{"TriggerLevel"};
static const QString bpp{"BytesPerPoint"};
static const QString bo{"ByteOrder"};
static const QString sRate{"SampleRate"};
static const QString recLen{"RecordLength"};
static const QString blockAvg{"BlockAverageEnabled"};
static const QString numAvg{"BlockAverages"};
static const QString multiRec{"MultiRecordEnabled"};
static const QString multiRecNum{"MultiRecordNum"};
}

class DigitizerConfig : public HeaderStorage
{
    Q_GADGET
public:
    struct AnalogChannel {
        double fullScale{0.0};
        double offset{0.0};
    };

    struct DigitalChannel {
        bool input{true};
        int role{-1};
    };

    enum TriggerSlope {
        RisingEdge,
        FallingEdge
    };
    Q_ENUM(TriggerSlope)

    enum ByteOrder {
        BigEndian = QDataStream::BigEndian,
        LittleEndian = QDataStream::LittleEndian
    };
    Q_ENUM(ByteOrder)

    DigitizerConfig(const QString key,const QString subKey);

    //vertical channels
    std::map<int,AnalogChannel> d_analogChannels;
    std::map<int,DigitalChannel> d_digitalChannels;

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
    ByteOrder d_byteOrder{LittleEndian};

    //averaging
    bool d_blockAverage{false};
    int d_numAverages{0};

    //multiRecord/fastframe
    bool d_multiRecord{false};
    int d_numRecords{0};

    double xIncr() const;
    double yMult(int ch) const;



    // HeaderStorage interface
protected:
    virtual void storeValues() override;
    virtual void retrieveValues() override;
};

Q_DECLARE_TYPEINFO(DigitizerConfig::AnalogChannel,Q_PRIMITIVE_TYPE);
Q_DECLARE_TYPEINFO(DigitizerConfig::DigitalChannel,Q_PRIMITIVE_TYPE);

#endif // DIGITIZERCONFIG_H
