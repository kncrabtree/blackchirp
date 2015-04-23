#ifndef LIFCONFIG_H
#define LIFCONFIG_H

#include <QSharedDataPointer>

#include <QDataStream>
#include <QList>
#include <QVector>
#include <QPointF>
#include <QMap>
#include <QVariant>

class LifConfigData;

class LifConfig
{
public:
    LifConfig();
    LifConfig(const LifConfig &);
    LifConfig &operator=(const LifConfig &);
    ~LifConfig();

    struct LifPoint {
        double mean;
        double sumsq;
        quint64 count;
    };

    enum LifScopeTriggerSlope {
        RisingEdge,
        FallingEdge
    };

    struct LifScopeConfig {
        double sampleRate;
        double vScale;
        int recordLength;
        int lifChannel;
        int triggerChannel;
        LifScopeTriggerSlope slope;

        int bytesPerPoint;
        QDataStream::ByteOrder byteOrder;
        double vOffset;
        double yMult;
        int yOff;
        double xIncr;

        LifScopeConfig() : sampleRate(0.0), vScale(0.0), recordLength(0), lifChannel(0), triggerChannel(0),
            slope(RisingEdge), bytesPerPoint(1), byteOrder(QDataStream::LittleEndian), vOffset(0.0),
            yMult(0.0), yOff(0), xIncr(0.0) {}

        QMap<QString,QPair<QVariant,QString> > headerMap() const
        {
            QMap<QString,QPair<QVariant,QString> > out;
            QString empty = QString("");
            QString prefix = QString("LifScope");
            QString scratch;

            out.insert(prefix+QString("LifChannel"),qMakePair(lifChannel,empty));
            out.insert(prefix+QString("VerticalScale"),qMakePair(QString::number(vScale,'f',3),QString("V/div")));
            out.insert(prefix+QString("VerticalOffset"),qMakePair(QString::number(vOffset,'f',3),QString("V")));
            out.insert(prefix+QString("TriggerChannel"),qMakePair(triggerChannel,empty));
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

    enum ScanOrder {
        DelayFirst,
        FrequencyFirst
    };

    bool isEnabled() const;
    bool isComplete() const;
    double currentDelay() const;
    double currentFrequency() const;
    int numDelayPoints() const;
    int numFrequencyPoints() const;
    int totalShots() const;
    int completedShots() const;
    QVector<QPointF> timeSlice(int frequencyIndex) const;
    QVector<QPointF> spectrum(int delayIndex) const;

    void setEnabled();
    bool addWaveform(const QByteArray b);


private:
    QSharedDataPointer<LifConfigData> data;

    bool addPoint(const double d);
    void increment();
};


class LifConfigData : public QSharedData
{
public:
    LifConfigData() : enabled(false), complete(false),  valid(false), order(LifConfig::DelayFirst),
        currentDelayIndex(0), currentFrequencyIndex(0) {}

    bool enabled;
    bool complete;
    bool valid;
    LifConfig::ScanOrder order;
    double delayStartUs;
    double delayEndUs;
    double delayStepUs;
    double frequencyStart;
    double frequencyEnd;
    double frequencyStep;
    int gateStartPoint;
    int gateEndPoint;
    int currentDelayIndex;
    int currentFrequencyIndex;
    int shotsPerPoint;

    LifConfig::LifScopeConfig scopeConfig;
    QList<QVector<LifConfig::LifPoint>> lifData;

};

#endif // LIFCONFIG_H
