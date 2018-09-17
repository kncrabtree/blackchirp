#ifndef DATASTRUCTS_H
#define DATASTRUCTS_H

#include <QMap>
#include <QString>
#include <QVariant>
#include <QMetaType>
#include <QDataStream>
#include <QSettings>
#include <QApplication>

namespace BlackChirp {

enum Sideband {
    UpperSideband,
    LowerSideband
};

enum LogMessageCode {
    LogNormal,
    LogWarning,
    LogError,
    LogHighlight,
    LogDebug
};

enum ScopeTriggerSlope {
    RisingEdge,
    FallingEdge
};

enum FtmwType
{
    FtmwTargetShots,
    FtmwTargetTime,
    FtmwForever,
    FtmwPeakUp
};

enum FtmwViewMode {
    FtmwViewLive,
    FtmwViewSingle,
    FtmwViewFrameDiff,
    FtmwViewSnapDiff
};

enum LifScanOrder {
    LifOrderDelayFirst,
    LifOrderFrequencyFirst
};

enum LifCompleteMode {
    LifStopWhenComplete,
    LifContinueUntilExperimentComplete
};


enum FlowSetting {
    FlowSettingEnabled,
    FlowSettingSetpoint,
    FlowSettingFlow,
    FlowSettingName
};

enum MotorAxis {
    MotorX,
    MotorY,
    MotorZ,
    MotorT
};

enum PulseActiveLevel { PulseLevelActiveLow, PulseLevelActiveHigh };
enum PulseSetting { PulseDelay, PulseWidth, PulseEnabled, PulseLevel, PulseName };

enum FtWindowFunction {
    Bartlett,
    Boxcar,
    Blackman,
    BlackmanHarris,
    Hamming,
    Hanning,
    KaiserBessel14
};

//NOTE: be sure to edit the functions below if adding a new clock type
enum ClockType {
    UpConversionLO,
    DownConversionLO,
    AwgClock,
    DRClock,
    DigitizerClock,
    ReferenceClock
};
inline bool operator <(ClockType &t1, ClockType &t2) { return static_cast<int>(t1) < static_cast<int>(t2); }

QList<ClockType> allClockTypes();
QString clockPrettyName(ClockType t);
QString clockKey(ClockType t);
BlackChirp::ClockType clockType(QString key);


struct FtmwScopeConfig {
    //user-chosen settings
    int fidChannel;
    double vScale;
    double sampleRate;
    int recordLength;
    bool fastFrameEnabled;
    int numFrames;
    bool summaryFrame;
    bool manualFrameAverage;
    bool blockAverageEnabled;
    bool blockAverageMultiply;//if device internally averages instead of sums, multiply to get approx raw ADC sum
    int numAverages;
    int trigChannel;
    double trigDelay; //in seconds
    double trigLevel; //in V
    ScopeTriggerSlope slope;

    //settings hardcoded or read from scope
    int bytesPerPoint;
    QDataStream::ByteOrder byteOrder; // set to BigEndian
    double vOffset; // set to 0
    double yMult; // read from scope (multiplier for digitized levels)
    int yOff; // read from scope (location of y=0 in digitized levels)
    double xIncr; // read from scope (actual point spacing in seconds)


    FtmwScopeConfig() : fidChannel(0), vScale(0.0), sampleRate(0.0), recordLength(0), fastFrameEnabled(false), numFrames(0),
        summaryFrame(false), manualFrameAverage(false), blockAverageEnabled(false), blockAverageMultiply(false), numAverages(1), trigChannel(0), trigDelay(0), trigLevel(0.0),
        slope(RisingEdge), bytesPerPoint(1), byteOrder(QDataStream::LittleEndian),
        vOffset(0.0), yMult(0.0), yOff(0), xIncr(0.0) {}
    FtmwScopeConfig(const FtmwScopeConfig &other) : fidChannel(other.fidChannel), vScale(other.vScale), sampleRate(other.sampleRate),
        recordLength(other.recordLength), fastFrameEnabled(other.fastFrameEnabled), numFrames(other.numFrames),
        summaryFrame(other.summaryFrame),manualFrameAverage(other.manualFrameAverage), blockAverageEnabled(other.blockAverageEnabled), blockAverageMultiply(other.blockAverageMultiply), numAverages(other.numAverages), trigChannel(other.trigChannel),
        trigDelay(other.trigDelay), trigLevel(other.trigLevel), slope(other.slope), bytesPerPoint(other.bytesPerPoint), byteOrder(other.byteOrder),
        vOffset(other.vOffset), yMult(other.yMult), yOff(other.yOff), xIncr(other.xIncr) {}

    QMap<QString,QPair<QVariant,QString> > headerMap() const
    {
        QMap<QString,QPair<QVariant,QString> > out;
        QString empty = QString("");
        QString prefix = QString("FtmwScope");
        QString scratch;

        out.insert(prefix+QString("FidChannel"),qMakePair(fidChannel,empty));
        out.insert(prefix+QString("VerticalScale"),qMakePair(QString::number(vScale,'f',3),QString("V/div")));
        out.insert(prefix+QString("VerticalOffset"),qMakePair(QString::number(vOffset,'f',3),QString("V")));
        scratch = QString("AuxIn");
        if(trigChannel > 0)
            scratch = QString::number(trigChannel);
        out.insert(prefix+QString("TriggerChannel"),qMakePair(scratch,empty));
        out.insert(prefix+QString("TriggerDelay"),qMakePair(QString::number(trigDelay),QString("s")));
        out.insert(prefix+QString("TriggerLevel"),qMakePair(QString::number(trigLevel),QString("V")));
        slope == RisingEdge ? scratch = QString("RisingEdge") : scratch = QString("FallingEdge");
        out.insert(prefix+QString("TriggerSlope"),qMakePair(scratch,empty));
        out.insert(prefix+QString("SampleRate"),qMakePair(QString::number(sampleRate/1e9,'f',3),QString("GS/s")));
        out.insert(prefix+QString("RecordLength"),qMakePair(recordLength,empty));
        out.insert(prefix+QString("FastFrame"),qMakePair(fastFrameEnabled,empty));
        out.insert(prefix+QString("NumFrames"),qMakePair(numFrames,empty));
        out.insert(prefix+QString("SummaryFrame"),qMakePair(summaryFrame,empty));
        out.insert(prefix+QString("BlockAverage"),qMakePair(blockAverageEnabled,empty));
        out.insert(prefix+QString("NumAverages"),qMakePair(numAverages,empty));
        out.insert(prefix+QString("BytesPerPoint"),qMakePair(bytesPerPoint,empty));
        byteOrder == QDataStream::BigEndian ? scratch = QString("BigEndian") : scratch = QString("LittleEndian");
        out.insert(prefix+QString("ByteOrder"),qMakePair(scratch,empty));

        return out;
    }
};

struct LifScopeConfig {
    double sampleRate;
    int recordLength;
    double xIncr;
    ScopeTriggerSlope slope;
    int bytesPerPoint;
    QDataStream::ByteOrder byteOrder;

    bool refEnabled;
    double vScale1, vScale2;
    double yMult1, yMult2;


    LifScopeConfig() : sampleRate(0.0), recordLength(0), xIncr(0.0), slope(RisingEdge), bytesPerPoint(1),
        byteOrder(QDataStream::LittleEndian), refEnabled(false), vScale1(0.0), vScale2(0.0), yMult1(0.0), yMult2(0.0) {}


    //Scope config
    QMap<QString,QPair<QVariant,QString> > headerMap() const
    {
        QMap<QString,QPair<QVariant,QString> > out;

        QString scratch;
        QString prefix = QString("LifScope");
        QString empty = QString("");

        out.insert(prefix+QString("LifVerticalScale"),qMakePair(QString::number(vScale1,'f',3),QString("V/div")));
        out.insert(prefix+QString("RefVerticalScale"),qMakePair(QString::number(vScale2,'f',3),QString("V/div")));
        slope == RisingEdge ? scratch = QString("RisingEdge") : scratch = QString("FallingEdge");
        out.insert(prefix+QString("TriggerSlope"),qMakePair(scratch,empty));
        out.insert(prefix+QString("SampleRate"),qMakePair(QString::number(sampleRate/1e9,'f',3),QString("GS/s")));
        out.insert(prefix+QString("RecordLength"),qMakePair(recordLength,empty));
        out.insert(prefix+QString("BytesPerPoint"),qMakePair(bytesPerPoint,empty));
        byteOrder == QDataStream::BigEndian ? scratch = QString("BigEndian") : scratch = QString("LittleEndian");
        out.insert(prefix+QString("ByteOrder"),qMakePair(scratch,empty));

        return out;
    }

};


struct LifPoint {
    double mean;
    double sumsq;
    quint64 count;

    LifPoint() : mean(0.0), sumsq(0.0), count(0) {}    
};

struct MotorScopeConfig {
    int dataChannel;
    double verticalScale;
    int recordLength;
    double sampleRate;
    int triggerChannel;
    BlackChirp::ScopeTriggerSlope slope;
    QDataStream::ByteOrder byteOrder;
    int bytesPerPoint;

    QMap<QString,QPair<QVariant,QString> > headerMap() const
    {
        QMap<QString,QPair<QVariant,QString> > out;

        QString scratch;
        QString prefix = QString("MotorScope");
        QString empty = QString("");

        out.insert(prefix+QString("VerticalScale"),qMakePair(QString::number(verticalScale,'f',3),QString("V/div")));
        slope == RisingEdge ? scratch = QString("RisingEdge") : scratch = QString("FallingEdge");
        out.insert(prefix+QString("TriggerSlope"),qMakePair(scratch,empty));
        out.insert(prefix+QString("SampleRate"),qMakePair(QString::number(sampleRate/1e6,'f',3),QString("MS/s")));
        out.insert(prefix+QString("RecordLength"),qMakePair(recordLength,empty));
        out.insert(prefix+QString("BytesPerPoint"),qMakePair(bytesPerPoint,empty));
        byteOrder == QDataStream::BigEndian ? scratch = QString("BigEndian") : scratch = QString("LittleEndian");
        out.insert(prefix+QString("ByteOrder"),qMakePair(scratch,empty));
        out.insert(prefix+QString("TriggerChannel"),qMakePair(QString::number(triggerChannel),empty));
        out.insert(prefix+QString("DataChannel"),qMakePair(QString::number(dataChannel),empty));

        return out;
    }

};


struct FlowChannelConfig {
    bool enabled;
    double setpoint;
    QString name;
};

struct ChirpSegment {
    double startFreqMHz;
    double endFreqMHz;
    double durationUs;
    double alphaUs;
    bool empty;
};

struct PulseChannelConfig {
    int channel;
    QString channelName;
    bool enabled;
    double delay;
    double width;
    PulseActiveLevel level;

    PulseChannelConfig() : channel(-1), enabled(false), delay(-1.0), width(-1.0), level(PulseLevelActiveHigh) {}
};

enum ExptFileType {
    HeaderFile,
    ChirpFile,
    FidFile,
    MultiFidFile,
    LifFile,
    SnapFile,
    TimeFile,
    LogFile,
    MotorFile
};

struct ValidationItem {
    QString key;
    double min;
    double max;
};

struct IOBoardChannel {
    bool enabled;
    QString name;
    bool plot;

    IOBoardChannel() {}
    IOBoardChannel(bool e, QString n, bool p) : enabled(e), name(n), plot(p) {}
    IOBoardChannel(const IOBoardChannel &other) : enabled(other.enabled), name(other.name), plot(other.plot) {}
};

enum FtPlotUnits {
    FtPlotV,
    FtPlotmV,
    FtPlotuV,
    FtPlotnV
};

QString getExptFile(int num, BlackChirp::ExptFileType t, QString path = QString(""), int snapNum = -1);
QString getExptDir(int num, QString path = QString(""));
QString getExportDir();
void setExportDir(const QString fileName);
QString headerMapToString(QMap<QString,QPair<QVariant,QString>> map);
QString channelNameLookup(QString key);


}

Q_DECLARE_METATYPE(BlackChirp::Sideband)
Q_DECLARE_METATYPE(BlackChirp::FlowSetting)
Q_DECLARE_METATYPE(BlackChirp::FtmwType)
Q_DECLARE_METATYPE(BlackChirp::ScopeTriggerSlope)
Q_DECLARE_METATYPE(BlackChirp::LifPoint)
Q_DECLARE_METATYPE(BlackChirp::LogMessageCode)
Q_DECLARE_METATYPE(BlackChirp::PulseActiveLevel)
Q_DECLARE_METATYPE(BlackChirp::LifScanOrder)
Q_DECLARE_METATYPE(BlackChirp::LifCompleteMode)
Q_DECLARE_METATYPE(BlackChirp::ValidationItem)
Q_DECLARE_METATYPE(BlackChirp::FtPlotUnits)
Q_DECLARE_METATYPE(BlackChirp::FtWindowFunction)
Q_DECLARE_METATYPE(BlackChirp::ClockType)

Q_DECLARE_TYPEINFO(BlackChirp::LifPoint,Q_MOVABLE_TYPE);
Q_DECLARE_TYPEINFO(BlackChirp::ChirpSegment,Q_MOVABLE_TYPE);
Q_DECLARE_TYPEINFO(BlackChirp::FlowChannelConfig,Q_MOVABLE_TYPE);
Q_DECLARE_TYPEINFO(BlackChirp::PulseChannelConfig,Q_MOVABLE_TYPE);
Q_DECLARE_TYPEINFO(BlackChirp::ValidationItem,Q_PRIMITIVE_TYPE);
Q_DECLARE_TYPEINFO(BlackChirp::IOBoardChannel,Q_PRIMITIVE_TYPE);

#endif // DATASTRUCTS_H

