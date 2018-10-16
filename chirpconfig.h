#ifndef CHIRPCONFIG_H
#define CHIRPCONFIG_H

#include <QSharedDataPointer>
#include <QPointF>
#include <QVector>
#include <QMap>
#include <QVariant>
#include <QPair>

#include "datastructs.h"

class ChirpConfigData;

//note: all time units are microseconds; all frequency units are MHz
class ChirpConfig
{
public:
    ChirpConfig();
    ChirpConfig(const ChirpConfig &);
    ChirpConfig &operator=(const ChirpConfig &);
    ChirpConfig(int num, QString path = QString(""));
    ~ChirpConfig();

    double preChirpProtectionDelay() const;
    double preChirpGateDelay() const;
    double postChirpGateDelay() const;
    double postChirpProtectionDelay() const;
    double totalProtectionWidth() const;
    double totalGateWidth() const;
    int numChirps() const;
    double chirpInterval() const;
    bool allChirpsIdentical() const;

    QList<QList<BlackChirp::ChirpSegment>> chirpList() const;
    double segmentStartFreq(int chirp, int segment) const;
    double segmentEndFreq(int chirp, int segment) const;
    double segmentDuration(int chirp, int segment) const;
    bool segmentEmpty(int chirp, int segment) const;
    QByteArray waveformHash() const;

    double chirpDuration(int chirpNum) const;
    double totalDuration() const;
    QVector<QPointF> getChirpMicroseconds() const;
    QVector<QPointF> getChirpSegmentMicroSeconds(double t1, double t2) const;
    QVector<QPair<bool,bool>> getMarkerData() const;
    QMap<QString,QPair<QVariant,QString>> headerMap() const;
    QString toString() const;

    void setAwgSampleRate(const double samplesPerSecond);
    void setPreChirpProtectionDelay(const double d);
    void setPreChirpGateDelay(const double d);
    void setPostChirpGateDelay(const double d);
    void setPostChirpProtectionDelay(const double d);
    void setNumChirps(const int n);
    void setChirpInterval(const double i);
    void addSegment(const double startMHz, const double endMHz, const double durationUs, const int chirpNum = -1);
    void addEmptySegment(const double durationUs, const int chirpNum = -1);
    void setChirpList(const QList<QList<BlackChirp::ChirpSegment> > l);

    void saveToSettings(int index) const;
    static ChirpConfig loadFromSettings(int index);

private:
    QSharedDataPointer<ChirpConfigData> data;

    int getFirstSample(double time) const;
    int getLastSample(double time) const;
    double getSampleTime(const int sample) const;
    double calculateChirp(const BlackChirp::ChirpSegment segment, const double t, const double phase) const;
    double calculateEndingPhaseRadians(const BlackChirp::ChirpSegment segment, const double endingTime, const double startingPhase) const;

    void parseFileLine(QByteArray line);
};

class ChirpConfigData : public QSharedData
{
public:
    //note: postChirpDelay initialized to 0.0 for backwards compatibility
    ChirpConfigData() : protectionDelaysUs(qMakePair(0.5,0.5)), gateDelaysUs(qMakePair(0.5,0.0)), chirpInterval(-1.0) {}

    QPair<double,double> protectionDelaysUs;
    QPair<double,double> gateDelaysUs;
    double chirpInterval; //units: us


    //working data to improve efficiency; do not record to disk!
    double sampleRateSperS; //awg rate, samples per second
    double sampleRateSperUS; //awg rate, samples per microecond
    double sampleIntervalS; //awg sample interval in seconds
    double sampleIntervalUS; //awg sample interval in microseconds

    QList<QList<BlackChirp::ChirpSegment>> chirpList;

};

#endif // CHIRPCONFIG_H
