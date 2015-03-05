#ifndef FTMWCONFIG_H
#define FTMWCONFIG_H

#include <QSharedDataPointer>
#include <QDateTime>
#include <QDataStream>
#include <QVariant>
#include "fid.h"

#ifdef BC_CUDA
namespace GpuAvg {
QString initializeAcquisition(const int bytesPerPoint, const int numPoints);
int gpuParseAndAdd(int bytesPerPoint, int numPoints, const char *newDataIn, long long int *sumData, bool littleEndian = true);
int acquisitionComplete();
}
#endif

class FtmwConfigData;

class FtmwConfig
{
public:
    FtmwConfig();
    FtmwConfig(const FtmwConfig &);
    FtmwConfig &operator=(const FtmwConfig &);
    ~FtmwConfig();

    enum ScopeTriggerSlope {
        RisingEdge,
        FallingEdge
    };

    struct ScopeConfig {
        //user-chosen settings
        int fidChannel;
        double vScale;
        double sampleRate;
        int recordLength;
        bool fastFrameEnabled;
        int numFrames;
        bool summaryFrame;
        int trigChannel;
        ScopeTriggerSlope slope;

        //settings hardcoded or read from scope
        int bytesPerPoint; // set to 2
        QDataStream::ByteOrder byteOrder; // set to BigEndian
        double vOffset; // set to 0
        double yMult; // read from scope (multiplier for digitized levels)
        int yOff; // read from scope (location of y=0 in digitized levels)
        double xIncr; // read from scope (actual point spacing in seconds)


        ScopeConfig() : fidChannel(0), vScale(0.0), sampleRate(0.0), recordLength(0), fastFrameEnabled(false), numFrames(0),
            summaryFrame(false), trigChannel(0), slope(RisingEdge), bytesPerPoint(1), byteOrder(QDataStream::LittleEndian),
            vOffset(0.0), yMult(0.0), yOff(0), xIncr(0.0) {}
        ScopeConfig(const ScopeConfig &other) : fidChannel(other.fidChannel), vScale(other.vScale), sampleRate(other.sampleRate),
            recordLength(other.recordLength), fastFrameEnabled(other.fastFrameEnabled), numFrames(other.numFrames),
            summaryFrame(other.summaryFrame), trigChannel(other.trigChannel), slope(other.slope), bytesPerPoint(other.bytesPerPoint),
            byteOrder(other.byteOrder), vOffset(other.vOffset), yMult(other.yMult), yOff(other.yOff), xIncr(other.xIncr) {}

        QHash<QString,QPair<QVariant,QString> > headerHash() const
        {
            QHash<QString,QPair<QVariant,QString> > out;
            QString empty = QString("");
            QString prefix = QString("FtmwScope");
            QString scratch;

            out.insert(prefix+QString("FidChannel"),qMakePair(fidChannel,empty));
            out.insert(prefix+QString("VerticalScale"),qMakePair(QString::number(vScale,'f',3),QString("V/div")));
            out.insert(prefix+QString("VerticalOffset"),qMakePair(QString::number(vOffset,'f',3),QString("V")));
            out.insert(prefix+QString("TriggerChannel"),qMakePair(trigChannel,empty));
            slope == RisingEdge ? scratch = QString("RisingEdge") : scratch = QString("FallingEdge");
            out.insert(prefix+QString("TriggerSlope"),qMakePair(scratch,empty));
            out.insert(prefix+QString("SampleRate"),qMakePair(QString::number(sampleRate/1e9,'f',3),QString("GS/s")));
            out.insert(prefix+QString("RecordLength"),qMakePair(recordLength,empty));
            out.insert(prefix+QString("FastFrame"),qMakePair(fastFrameEnabled,empty));
            out.insert(prefix+QString("NumFrames"),qMakePair(numFrames,empty));
            out.insert(prefix+QString("SummaryFrame"),qMakePair(summaryFrame,empty));
            out.insert(prefix+QString("BytesPerPoint"),qMakePair(bytesPerPoint,empty));
            byteOrder == QDataStream::BigEndian ? scratch = QString("BigEndian") : scratch = QString("LittleEndian");
            out.insert(prefix+QString("ByteOrder"),qMakePair(scratch,empty));

            return out;
        }

    };

    enum FtmwType
    {
        TargetShots,
        TargetTime,
        Forever,
        PeakUp
    };

    bool isEnabled() const;
    FtmwConfig::FtmwType type() const;
    qint64 targetShots() const;
    qint64 completedShots() const;
    QDateTime targetTime() const;
    int autoSaveShots() const;
    double loFreq() const;
    Fid::Sideband sideband() const;
    QList<Fid> fidList() const;
    ScopeConfig scopeConfig() const;
    Fid fidTemplate() const;
    int numFrames() const;
    QList<Fid> parseWaveform(QByteArray b) const;
    QString errorString() const;
    void finishAcquisition() const;

    bool prepareForAcquisition();
    void setEnabled();
    void setFidTemplate(const Fid f);
    void setType(const FtmwConfig::FtmwType type);
    void setTargetShots(const qint64 target);
    void increment();
    void setTargetTime(const QDateTime time);
    void setAutoSaveShots(const int shots);
    void setLoFreq(const double f);
    void setSideband(const Fid::Sideband sb);
    bool setFids(const QByteArray newData);
    bool addFids(const QByteArray rawData);
    void setScopeConfig(const ScopeConfig &other);


    bool isComplete() const;
    QHash<QString,QPair<QVariant,QString> > headerHash() const;

private:
    QSharedDataPointer<FtmwConfigData> data;

};

class FtmwConfigData : public QSharedData
{
public:
    FtmwConfigData() : isEnabled(false), type(FtmwConfig::Forever), targetShots(-1), completedShots(0), autoSaveShots(1000), loFreq(0.0), sideband(Fid::UpperSideband) {}

    bool isEnabled;
    FtmwConfig::FtmwType type;
    qint64 targetShots;
    qint64 completedShots;
    QDateTime targetTime;
    int autoSaveShots;

    double loFreq;
    Fid::Sideband sideband;
    QList<Fid> fidList;

    FtmwConfig::ScopeConfig scopeConfig;
    QVector<qint64> rawData;
    Fid fidTemplate;
    QString errorString;

};

Q_DECLARE_TYPEINFO(FtmwConfig, Q_MOVABLE_TYPE);


#endif // FTMWCONFIG_H
