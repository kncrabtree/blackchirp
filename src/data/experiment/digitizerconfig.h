#ifndef DIGITIZERCONFIG_H
#define DIGITIZERCONFIG_H

#include <QObject>
#include <QDataStream>
#include <set>

#include <data/storage/headerstorage.h>
#include <data/settings/hardwarekeys.h>

namespace BC::Store::Digi {
inline constexpr QLatin1StringView an{"AnalogChannel"};
inline constexpr QLatin1StringView dig{"DigitalChannel"};
inline constexpr QLatin1StringView chIndex{"Index"};
inline constexpr QLatin1StringView digInp{"Input"};
inline constexpr QLatin1StringView digRole{"Role"};
inline constexpr QLatin1StringView en{"Enabled"};
inline constexpr QLatin1StringView fs{"FullScale"};
inline constexpr QLatin1StringView offset{"VerticalOffset"};
inline constexpr QLatin1StringView trigSlope{"TriggerEdge"};
inline constexpr QLatin1StringView trigCh{"TriggerChannel"};
inline constexpr QLatin1StringView trigDelay{"TriggerDelay"};
inline constexpr QLatin1StringView trigLevel{"TriggerLevel"};
inline constexpr QLatin1StringView bpp{"BytesPerPoint"};
inline constexpr QLatin1StringView bo{"ByteOrder"};
inline constexpr QLatin1StringView sRate{"SampleRate"};
inline constexpr QLatin1StringView recLen{"RecordLength"};
inline constexpr QLatin1StringView blockAvg{"BlockAverageEnabled"};
inline constexpr QLatin1StringView numAvg{"BlockAverages"};
inline constexpr QLatin1StringView multiRec{"MultiRecordEnabled"};
inline constexpr QLatin1StringView multiRecNum{"MultiRecordNum"};
}

class DigitizerConfig : public HeaderStorage
{
    Q_GADGET
public:
    struct AnalogChannel {
        bool enabled{false};
        double fullScale{0.0};
        double offset{0.0};
    };

    struct DigitalChannel {
        bool enabled{false};
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

    DigitizerConfig(const QString& hwKey);

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
