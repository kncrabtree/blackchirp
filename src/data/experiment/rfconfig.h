#ifndef RFCONFIG_H
#define RFCONFIG_H

#include <QSharedDataPointer>

#include <data/storage/headerstorage.h>
#include <data/experiment/chirpconfig.h>

namespace BC::Store::RFC {
static const QString key("RfConfig");
static const QString commonLO("CommonUpDownLO");
static const QString targetSweeps("TargetSweeps");
static const QString shots("ShotsPerClockConfig");
static const QString awgM("AwgMult");
static const QString upSB("UpconversionSideband");
static const QString chirpM("ChirpMult");
static const QString downSB("DownconversionSideband");
}

/**
 * @brief Configuration for RF/Microwave sources
 *
 * The RfConfig class is designed to be a bridge between the
 * FtmwConfig class and the Chirp/Ramp configuration. This
 * helps to facilitate multiple acquisition types (e.g., segmented)
 */
class RfConfig : public HeaderStorage
{
    Q_GADGET
public:
    enum MultOperation {
        Multiply,
        Divide
    };
    Q_ENUM(MultOperation)

    enum Sideband {
        UpperSideband,
        LowerSideband
    };
    Q_ENUM(Sideband)

    enum ClockType {
        UpLO,
        DownLO,
        AwgRef,
        DRClock,
        DigRef,
        ComRef
    };
    Q_ENUM(ClockType)

    struct ClockFreq {
        double desiredFreqMHz{0.0};
        MultOperation op{Multiply};
        double factor{1.0};
        QString hwKey{""};
        int output{0};
    };

    RfConfig();
    ~RfConfig();

    //options
    bool d_commonUpDownLO{false};
    int d_completedSweeps{0};
    int d_targetSweeps{1};
    int d_shotsPerClockConfig{0};

    //Upconversion chain
    double d_awgMult{1.0};
    Sideband d_upMixSideband{UpperSideband};
    double d_chirpMult{1.0};

    //downconversion chain
    Sideband d_downMixSideband{UpperSideband};

    //chirp
    ChirpConfig d_chirpConfig;

    bool prepareForAcquisition();
    void setCurrentClocks(const QHash<ClockType,ClockFreq> clocks);
    void setClockDesiredFreq(ClockType t, double f);
    void setClockFreqInfo(ClockType t, const ClockFreq &cf);
    void addClockStep(QHash<ClockType,ClockFreq> h);
    void addLoScanClockStep(double upLoMHz, double downLoMHz);
    void addDrScanClockStep(double drFreqMHz);
    void clearClockSteps();
    void setChirpConfig(const ChirpConfig &cc);
    int advanceClockStep();

    quint64 totalShots() const;
    quint64 completedSegmentShots() const;
    bool canAdvance(qint64 shots) const;
    int numSegments() const;

    QVector<QHash<ClockType,RfConfig::ClockFreq>> clockSteps() const;
    QHash<ClockType,ClockFreq> getClocks() const;
    double clockFrequency(ClockType t) const;
    double rawClockFrequency(ClockType t) const;
    QString clockHardware(ClockType t) const;
    bool isComplete() const;

    double calculateChirpFreq(double awgFreq) const;
    double calculateAwgFreq(double chirpFreq) const;
    double calculateChirpAbsOffset(double awgFreq) const;
    QPair<double,double> calculateChirpAbsOffsetRange() const;

    bool writeClockFile(int num) const;
    void loadClockSteps(BlackchirpCSV *csv, int num, QString path);

private:
    //clocks
    QHash<ClockType,ClockFreq> d_clockTemplate;
    QVector<QHash<ClockType,RfConfig::ClockFreq>> d_clockConfigs;
    int d_currentClockIndex{-1};

    double getRawFrequency(ClockFreq f) const;

    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
    void prepareChildren() override;
};

Q_DECLARE_METATYPE(RfConfig)
Q_DECLARE_METATYPE(RfConfig::MultOperation)
Q_DECLARE_METATYPE(RfConfig::ClockType)
Q_DECLARE_METATYPE(RfConfig::ClockFreq)

#endif // RFCONFIG_H
