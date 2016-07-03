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

    bool compareTxParams(const ChirpConfig &other) const;
    bool isValid() const;
    double preChirpProtection() const;
    double preChirpDelay() const;
    double postChirpProtection() const;
    int numChirps() const;
    double chirpInterval() const;

    QList<BlackChirp::ChirpSegment> segmentList() const;
    double segmentStartFreq(int i) const;
    double segmentEndFreq(int i) const;
    double segmentDuration(int i) const;
    QByteArray waveformHash() const;

    double chirpDuration() const;
    double totalDuration() const;
    QVector<QPointF> getChirpMicroseconds() const;
    QVector<QPointF> getChirpSegmentMicroSeconds(double t1, double t2) const;
    QVector<QPair<bool,bool>> getMarkerData() const;
    QMap<QString,QPair<QVariant,QString>> headerMap() const;
    QString toString() const;

    double synthTxMult() const;
    double awgMult() const;
    double mixerSideband() const;
    double totalMult() const;
    double synthTxFreq() const;

    void setPreChirpProtection(const double d);
    void setPreChirpDelay(const double d);
    void setPostChirpProtection(const double d);
    void setNumChirps(const int n);
    void setChirpInterval(const double i);
    void addSegment(const double startMHz, const double endMHz, const double durationUs);
    void addEmptySegment(const double durationUs);
    void setSegmentList(const QList<BlackChirp::ChirpSegment> l);

    void setTxFreq(double f);
    void setTxMult(double m);
    void setAwgMult(double m);
    void setTxSideband(double s);
    void setTotalMult(double m);

    void saveToSettings() const;
    static ChirpConfig loadFromSettings();

private:
    QSharedDataPointer<ChirpConfigData> data;

    int getFirstSample(double time) const;
    int getLastSample(double time) const;
    double getSampleTime(const int sample) const;
    double calculateChirp(const BlackChirp::ChirpSegment segment, const double t, const double phase) const;
    double calculateEndingPhaseRadians(const BlackChirp::ChirpSegment segment, const double endingTime, const double startingPhase) const;
    double realToAwgFreq(const double realFreq) const;
    double awgToRealFreq(const double awgFreq) const;

    bool validate();
    void parseFileLine(QByteArray line);
};

class ChirpConfigData : public QSharedData
{
public:
    ChirpConfigData() : preChirpProtection(-1.0), preChirpDelay(-1.0), postChirpProtection(-1.0), numChirps(0), chirpInterval(-1.0),
        synthTxFreq(-1.0), synthTxMult(-1.0), awgMult(-1.0), mixerTxSideband(0.0), totalMult(0.0), isValid(false) {}

    double preChirpProtection; //units: us
    double preChirpDelay; //units: us
    double postChirpProtection; //units: us
    int numChirps;
    double chirpInterval; //units: us

    double synthTxFreq; //units: MHz
    double synthTxMult;
    double awgMult;
    double mixerTxSideband; //+1.0 or -1.0
    double totalMult;


    //working data to improve efficiency; do not record to disk!
    double sampleRateSperS; //awg rate, samples per second
    double sampleRateSperUS; //awg rate, samples per microecond
    double sampleIntervalS; //awg sample interval in seconds
    double sampleIntervalUS; //awg sample interval in microseconds
    bool isValid;

    QList<BlackChirp::ChirpSegment> segments;

};

#endif // CHIRPCONFIG_H
