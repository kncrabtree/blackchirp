#ifndef OSCILLOSCOPE_H
#define OSCILLOSCOPE_H

#include "tcpinstrument.h"
#include <QVector>
#include <QPointF>
#include <QDataStream>
#include <QTextStream>
#include <QPointF>
#include <QTime>
#include <QStringList>
#include "fid.h"
#include <QTimer>

class Oscilloscope : public TcpInstrument
{
    Q_OBJECT
public:
    explicit Oscilloscope(QObject *parent = 0);

    enum ResponseType {
        RawData,
        BlockData
    };

    enum TriggerSlope {
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
        TriggerSlope slope;

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

//        QStringList labels() const {
//            QStringList out;
//            out.append(QString("FidChannel"));
//            out.append(QString("VerticalScale"));
//            out.append(QString("VerticalOffset"));
//            out.append(QString("TriggerChannel"));
//            out.append(QString("TriggerSlope"));
//            out.append(QString("SampleRate"));
//            out.append(QString("RecordLength"));
//            out.append(QString("FastFrame"));
//            out.append(QString("NumFrames"));
//            out.append(QString("SummaryFrame"));
//            out.append(QString("BytesPerPoint"));
//            out.append(QString("ByteOrder"));
//            return out;
//        }
//        QList<QVariant> values() const {
//            QList<QVariant> out;
//            out.append(fidChannel);
//            out.append();
//            out.append(QString::number(vOffset,'f',3));
//            out.append(trigChannel);
//            slope == RisingEdge ? out.append(QString("RisingEdge")) : out.append(QString("FallingEdge"));
//            out.append(QString::number(sampleRate/1e9,'f',3));
//            out.append(recordLength);
//            out.append(fastFrameEnabled);
//            out.append(numFrames);
//            out.append(summaryFrame);
//            out.append(bytesPerPoint);
//            byteOrder == QDataStream::BigEndian ? out.append(QString("BigEndian")) : out.append(QString("LittleEndian"));
//            return out;
//        }
//        QStringList units() const {
//            QStringList out;

//            out.append(empty);
//            out.append(QString("V/div"));
//            out.append(QString("V"));
//            out.append(empty);
//            out.append(empty);
//            out.append(QString("GS/s"));
//            out.append(QString("pts"));
//            out.append(empty);
//            out.append(empty);
//            out.append(empty);
//            out.append(empty);
//            out.append(empty);
//            return out;
//        }
    };

    static Fid parseWaveform(QByteArray b, const ScopeConfig &config, const double loFreq, const Fid::Sideband sb);

signals:
    void shotAcquired(const quint64 id, const QByteArray data);

public slots:
    void initialize();
    bool testConnection();

    void readWaveform();
    void endAcquisition(bool unlock = true);

    Oscilloscope::ScopeConfig initializeAcquisition(const Oscilloscope::ScopeConfig &config);
    void beginAcquisition();
    void queryScope(quint64 id);
    void wakeUp();

private:
    quint64 d_shotId;
    bool d_waitingForReply;
    bool d_foundHeader;
    int d_headerNumBytes;
    int d_waveformBytes;
    ScopeConfig d_configuration;
    QDateTime d_lastTrigger;
    bool d_waitingForWakeUp;
    QTimer d_scopeTimeout;

    QByteArray scopeQueryCmd(QString query);

    QByteArray makeSimulatedData();
    QVector<double> d_simulatedData;




};

Q_DECLARE_METATYPE(Oscilloscope::ScopeConfig)

#endif // OSCILLOSCOPE_H
