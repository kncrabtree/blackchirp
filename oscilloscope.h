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

        QStringList labels() const {
            QStringList out;
            out.append(QString("FID channel"));
            out.append(QString("Vertical scale"));
            out.append(QString("Vertical offset"));
            out.append(QString("Trigger channel"));
            out.append(QString("Trigger slope"));
            out.append(QString("Sample rate"));
            out.append(QString("Record length"));
            out.append(QString("Fast frame"));
            out.append(QString("Num frames"));
            out.append(QString("Summary frame"));
            out.append(QString("Bytes per point"));
            out.append(QString("Byte order"));
            return out;
        }
        QStringList values() const {
            QStringList out;
            out.append(QString::number(fidChannel));
            out.append(QString::number(vScale,'f',3));
            out.append(QString::number(vOffset,'f',3));
            out.append(QString::number(trigChannel));
            slope == RisingEdge ? out.append(QString("Rising edge")) : out.append(QString("Falling edge"));
            out.append(QString::number(sampleRate/1e9,'f',3));
            out.append(QString::number(recordLength));
            fastFrameEnabled ? out.append(QString("Yes")) : out.append(QString("No"));
            out.append(QString::number(numFrames));
            summaryFrame ? out.append(QString("Yes")) : out.append(QString("No"));
            out.append(QString::number(bytesPerPoint));
            byteOrder == QDataStream::BigEndian ? out.append(QString("Big endian")) : out.append(QString("Little endian"));
            return out;
        }
        QStringList units() const {
            QStringList out;
            out.append(QString(""));
            out.append(QString("V/div"));
            out.append(QString("V"));
            out.append(QString(""));
            out.append(QString(""));
            out.append(QString("GS/s"));
            out.append(QString("pts"));
            out.append(QString(""));
            out.append(QString(""));
            out.append(QString(""));
            out.append(QString(""));
            out.append(QString(""));
            return out;
        }
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
