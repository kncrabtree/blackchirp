#ifndef LIFCONFIG_H
#define LIFCONFIG_H

#include <QSharedDataPointer>

#include <QDataStream>
#include <QList>
#include <QVector>
#include <QPointF>
#include <QMap>
#include <QVariant>

#include "liftrace.h"

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

        LifPoint() : mean(0.0), sumsq(0.0), count(0) {}
    };

    enum LifScopeTriggerSlope {
        RisingEdge,
        FallingEdge
    };

    struct LifScopeConfig {
        double sampleRate;
        int recordLength;
        double xIncr;
        LifScopeTriggerSlope slope;
        int bytesPerPoint;
        QDataStream::ByteOrder byteOrder;

        double vScale1, vScale2;
        double yMult1, yMult2;


        LifScopeConfig() : sampleRate(0.0), recordLength(0), xIncr(0.0), slope(RisingEdge), bytesPerPoint(1),
            byteOrder(QDataStream::LittleEndian), vScale1(0.0), vScale2(0.0), yMult1(0.0), yMult2(0.0) {}

        QMap<QString,QPair<QVariant,QString> > headerMap() const
        {
            QMap<QString,QPair<QVariant,QString> > out;
            QString empty = QString("");
            QString prefix = QString("LifScope");
            QString scratch;

            out.insert(prefix+QString("VerticalScale1"),qMakePair(QString::number(vScale1,'f',3),QString("V/div")));
            out.insert(prefix+QString("VerticalScale2"),qMakePair(QString::number(vScale2,'f',3),QString("V/div")));
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
    LifTrace parseWaveform(const QByteArray b) const;
    QMap<QString,QPair<QVariant,QString> > headerMap() const;
    QPoint lastUpdatedPointIndices() const;
    LifPoint lastUpdatedLifPoint() const;

    bool setEnabled();
    bool validate();
    void setLifGate(int start, int end);
    void setRefGate(int start, int end);
    bool addWaveform(const LifTrace t);


private:
    QSharedDataPointer<LifConfigData> data;

    bool addPoint(const double d);
    void increment();
};


class LifConfigData : public QSharedData
{
public:
    LifConfigData() : enabled(false), complete(false),  valid(false), refEnabled(false), order(LifConfig::DelayFirst),
        delayStartUs(-1.0), delayEndUs(-1.0), delayStepUs(0.0), frequencyStart(-1.0), frequencyEnd(-1.0),
        frequencyStep(0.0), lifGateStartPoint(-1), lifGateEndPoint(-1),
        refGateStartPoint(-1), refGateEndPoint(-1), currentDelayIndex(0), currentFrequencyIndex(0) {}

    bool enabled;
    bool complete;
    bool valid;
    bool refEnabled;
    LifConfig::ScanOrder order;
    double delayStartUs;
    double delayEndUs;
    double delayStepUs;
    double frequencyStart;
    double frequencyEnd;
    double frequencyStep;
    int lifGateStartPoint;
    int lifGateEndPoint;
    int refGateStartPoint;
    int refGateEndPoint;
    int currentDelayIndex;
    int currentFrequencyIndex;
    int shotsPerPoint;
    QPoint lastUpdatedPoint;

    LifConfig::LifScopeConfig scopeConfig;
    QList<QVector<LifConfig::LifPoint>> lifData;

};

Q_DECLARE_METATYPE(LifConfig::LifScopeConfig)
Q_DECLARE_METATYPE(LifConfig::LifPoint)
Q_DECLARE_METATYPE(LifConfig)
Q_DECLARE_TYPEINFO(LifConfig::LifPoint,Q_MOVABLE_TYPE);

#endif // LIFCONFIG_H
