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

enum LogMessageCode {
    LogNormal,
    LogWarning,
    LogError,
    LogHighlight,
    LogDebug
};

enum ScopeSampleOrder {
    ChannelsSequential,
    ChannelsInterleaved
};

#ifdef BC_LIF

enum LifScanOrder {
    LifOrderDelayFirst,
    LifOrderFrequencyFirst
};

enum LifCompleteMode {
    LifStopWhenComplete,
    LifContinueUntilExperimentComplete
};

struct LifScopeConfig {
    double sampleRate;
    int recordLength;
    double xIncr;
    ScopeTriggerSlope slope;
    int bytesPerPoint;
    DigitizerConfig::ByteOrder byteOrder;
    ScopeSampleOrder channelOrder;

    bool refEnabled;
    double vScale1, vScale2;
    double yMult1, yMult2;


    LifScopeConfig() : sampleRate(0.0), recordLength(0), xIncr(0.0), slope(RisingEdge), bytesPerPoint(1),
        byteOrder(DigitizerConfig::LittleEndian), channelOrder(ChannelsInterleaved), refEnabled(false), vScale1(0.0),
        vScale2(0.0), yMult1(0.0), yMult2(0.0) {}


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
        byteOrder == DigitizerConfig::BigEndian ? scratch = QString("BigEndian") : scratch = QString("LittleEndian");
        out.insert(prefix+QString("ByteOrder"),qMakePair(scratch,empty));
        channelOrder == ChannelsInterleaved ? scratch = QString("Interleaved") : scratch = QString("Sequential");
        out.insert(prefix+QString("ChannelOrder"),qMakePair(scratch,empty));

        return out;
    }

};

#endif

#ifdef BC_MOTOR

struct MotorScopeConfig {
    int dataChannel;
    double verticalScale;
    int recordLength;
    double sampleRate;
    int triggerChannel;
    BlackChirp::ScopeTriggerSlope slope;
    DigitizerConfig::ByteOrder byteOrder;
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
        byteOrder == DigitizerConfig::BigEndian ? scratch = QString("BigEndian") : scratch = QString("LittleEndian");
        out.insert(prefix+QString("ByteOrder"),qMakePair(scratch,empty));
        out.insert(prefix+QString("TriggerChannel"),qMakePair(QString::number(triggerChannel),empty));
        out.insert(prefix+QString("DataChannel"),qMakePair(QString::number(dataChannel),empty));

        return out;
    }

};
#endif


}

Q_DECLARE_METATYPE(BlackChirp::LogMessageCode)

#ifdef BC_LIF
Q_DECLARE_METATYPE(BlackChirp::LifScanOrder)
Q_DECLARE_METATYPE(BlackChirp::LifCompleteMode)
#endif

#endif // DATASTRUCTS_H

