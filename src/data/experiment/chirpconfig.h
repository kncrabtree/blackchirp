#ifndef CHIRPCONFIG_H
#define CHIRPCONFIG_H

#include <QtGlobal>
#include <QPointF>
#include <QVector>
#include <QMap>
#include <QVariant>
#include <QPair>

#include <data/storage/headerstorage.h>

namespace BC::Store::CC {
static const QString key("ChirpConfig");
static const QString preProt("PreProtection");
static const QString postProt("PostProtection");
static const QString preGate("PreGate");
static const QString postGate("PostGate");
static const QString interval("ChirpInterval");
static const QString sampleRate("SampleRate");
static const QString sampleInterval("SampleInterval");
}

class BlackchirpCSV;

//note: all time units are microseconds; all frequency units are MHz
class ChirpConfig : public HeaderStorage
{
public:
    struct ChirpSegment {
        double startFreqMHz;
        double endFreqMHz;
        double durationUs;
        double alphaUs;
        bool empty;
    };

    ChirpConfig();
    ~ChirpConfig();

    void readChirpFile(BlackchirpCSV *csv, int num, QString path = QString(""));
    bool writeChirpFile(int num) const;

    double preChirpProtectionDelay() const;
    double preChirpGateDelay() const;
    double postChirpGateDelay() const;
    double postChirpProtectionDelay() const;
    double totalProtectionWidth() const;
    double totalGateWidth() const;
    int numChirps() const;
    double chirpInterval() const;
    bool allChirpsIdentical() const;

    QVector<QVector<ChirpSegment>> chirpList() const;
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

    void setAwgSampleRate(const double samplesPerSecond);
    void setPreChirpProtectionDelay(const double d);
    void setPreChirpGateDelay(const double d);
    void setPostChirpGateDelay(const double d);
    void setPostChirpProtectionDelay(const double d);
    void setNumChirps(const int n);
    void setChirpInterval(const double i);
    void addSegment(const double startMHz, const double endMHz, const double durationUs, const int chirpNum = -1);
    void addEmptySegment(const double durationUs, const int chirpNum = -1);
    void setChirpList(const QVector<QVector<ChirpSegment> > l);

private:
    struct Markers {
        double preProt{0.5};
        double postProt{0.5};
        double preGate{0.5};
        double postGate{0.5};
    } d_markers;

    double d_chirpInterval{-1.0}; //units: us


    //working data to improve efficiency; do not record to disk!
    double d_sampleRateSperUS; //awg rate, samples per microecond
    double d_sampleIntervalUS; //awg sample interval in microseconds

    int getFirstSample(double time) const;
    int getLastSample(double time) const;
    double getSampleTime(const int sample) const;
    double calculateChirp(const ChirpSegment segment, const double t, const double phase) const;
    double calculateEndingPhaseRadians(const ChirpSegment segment, const double endingTime, const double startingPhase) const;

    QVector<QVector<ChirpSegment>> d_chirpList;


    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
};

Q_DECLARE_TYPEINFO(ChirpConfig::ChirpSegment,Q_PRIMITIVE_TYPE);

#endif // CHIRPCONFIG_H
