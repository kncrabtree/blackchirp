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
        double desiredFreqMHz;
        MultOperation op;
        double factor;
        QString hwKey;
        int output;
    };

    RfConfig();
    ~RfConfig();

    //options
    bool d_commonUpDownLO{false};
    int d_targetSweeps{1};
    int d_shotsPerClockConfig{0};

    //Upconversion chain
    double d_awgMult{1.0};
    Sideband d_upMixSideband{UpperSideband};
    double d_chirpMult{1.0};

    //downconversion chain
    Sideband d_downMixSideband{UpperSideband};

    bool prepareForAcquisition();
    void setClockDesiredFreq(ClockType t, double targetFreqMHz);
    void setClockFactor(ClockType t, double factor);
    void setClockOp(ClockType t, MultOperation o);
    void setClockOutputNum(ClockType t, int output);
    void setClockHwKey(ClockType t, QString key);
    void setClockHwInfo(ClockType t, QString hwKey, int output);
    void setClockFreqInfo(ClockType t, double targetFreqMHz = 0.0, double factor = 1.0, MultOperation o = Multiply, QString hwKey = QString(""), int output = 0);
    void setClockFreqInfo(ClockType t, const ClockFreq &cf);
    void addClockStep(QHash<ClockType,ClockFreq> h);
    void addLoScanClockStep(double upLoMHz, double downLoMHz);
    void addDrScanClockStep(double drFreqMHz);
    void clearClockSteps();
    void clearChirpConfigs();
    bool setChirpConfig(const ChirpConfig cc, int num=0);
    void addChirpConfig(ChirpConfig cc);
    int advanceClockStep();


    int completedSweeps() const;
    quint64 totalShots() const;
    quint64 completedSegmentShots() const;
    bool canAdvance(qint64 shots) const;
    int numSegments() const;

    QHash<ClockType,ClockFreq> getClocks() const;
    double clockFrequency(ClockType t) const;
    double rawClockFrequency(ClockType t) const;
    QString clockHardware(ClockType t) const;
    ChirpConfig getChirpConfig(int num=0) const;
    int numChirpConfigs() const;
    bool isComplete() const;

    double calculateChirpFreq(double awgFreq) const;
    double calculateAwgFreq(double chirpFreq) const;
    double calculateChirpAbsOffset(double awgFreq) const;
    QPair<double,double> calculateChirpAbsOffsetRange() const;

    bool writeClockFile(int num, QString path) const;
    void loadClockSteps(int num, QString path);

private:
    //clocks
    QHash<ClockType,ClockFreq> d_clockTemplate;
    QVector<QHash<ClockType,RfConfig::ClockFreq>> d_clockConfigs;
    int d_currentClockIndex{0};
    int d_completedSweeps{0};

    //chirps
    QVector<ChirpConfig> d_chirps;

    double getRawFrequency(ClockFreq f) const;

    // HeaderStorage interface
protected:
    void prepareToSave() override;
    void loadComplete() override;
};

Q_DECLARE_METATYPE(RfConfig)
Q_DECLARE_METATYPE(RfConfig::MultOperation)

#endif // RFCONFIG_H
