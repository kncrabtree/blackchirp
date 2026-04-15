#ifndef CHIRPCONFIG_H
#define CHIRPCONFIG_H

#include <QtGlobal>
#include <QPointF>
#include <QVector>
#include <QMap>
#include <QVariant>
#include <QString>

#include <data/storage/headerstorage.h>

namespace BC::Store::CC {
inline constexpr QLatin1StringView key{"ChirpConfig"};
inline constexpr QLatin1StringView interval{"ChirpInterval"};
inline constexpr QLatin1StringView sampleRate{"SampleRate"};
inline constexpr QLatin1StringView sampleInterval{"SampleInterval"};
}

enum class MarkerRole { Protection, Gate, Trigger, Custom };

struct MarkerChannel {
    QString name;
    enum TimingMode { Absolute, ChirpRelative };
    TimingMode timingMode{ChirpRelative};
    double startTime{-0.5};  // us -- relative to chirp start (negative = before)
    double endTime{0.5};     // us -- relative to chirp end (positive = after)
    bool enabled{true};
    MarkerRole role{MarkerRole::Custom};
};

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
    void readMarkersFile(BlackchirpCSV *csv, int num, QString path = QString(""));
    bool writeMarkersFile(int num) const;

    double leadTimeUs() const;
    double tailTimeUs() const;
    int numChirps() const;
    double chirpInterval() const;
    bool allChirpsIdentical() const;

    QVector<QVector<ChirpSegment>> chirpList() const;
    double segmentStartFreq(int chirp, int segment) const;
    double segmentEndFreq(int chirp, int segment) const;
    double segmentDuration(int chirp, int segment) const;
    bool segmentEmpty(int chirp, int segment) const;
    QByteArray waveformHash() const;

    double chirpDurationUs(int chirpNum) const;
    double totalDuration() const;
    QVector<QPointF> getChirpMicroseconds() const;
    QVector<QPointF> getChirpSegmentMicroSeconds(double t1, double t2) const;
    QVector<QVector<bool>> getMarkerData() const;
    QVector<quint32> getPackedMarkerData() const;

    const QVector<MarkerChannel>& markerChannels() const;
    const MarkerChannel* findEnabledMarkerByRole(MarkerRole role) const;

    void setAwgSampleRate(const double samplesPerSecond);
    void setNumChirps(const int n);
    void setChirpInterval(const double i);
    void setMarkerChannels(const QVector<MarkerChannel>& channels);
    void addSegment(const double startMHz, const double endMHz, const double durationUs, const int chirpNum = -1);
    void addEmptySegment(const double durationUs, const int chirpNum = -1);
    void setChirpList(const QVector<QVector<ChirpSegment> > l);

private:
    QVector<MarkerChannel> d_markerChannels;
    double d_chirpInterval{-1.0}; //units: us


    //working data to improve efficiency; do not record to disk!
    double d_sampleRateSperUS; //awg rate, samples per microsecond
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
