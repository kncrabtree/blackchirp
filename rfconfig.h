#ifndef RFCONFIG_H
#define RFCONFIG_H

#include <QSharedDataPointer>

#include "datastructs.h"
#include "chirpconfig.h"

class RfConfigData;

///TODO: Write .rfc file!
///Might make sense to wait for the big BlackChirpDataStorage refactor though!

/**
 * @brief Configuration for RF/Microwave sources
 *
 * The RfConfig class is designed to be a bridge between the
 * FtmwConfig class and the Chirp/Ramp configuration. This
 * helps to facilitate multiple acquisition types (e.g., segmented)
 */
class RfConfig
{
public:

    enum MultOperation {
        Multiply,
        Divide
    };

    struct ClockFreq {
        double desiredFreqMHz;
        MultOperation op;
        double factor;
        QString hwKey;
        int output;
    };

    RfConfig();
    RfConfig(const RfConfig &);
    RfConfig &operator=(const RfConfig &);
    ~RfConfig();

    void saveToSettings() const;
    static RfConfig loadFromSettings();
    QMap<QString,QPair<QVariant,QString> > headerMap() const;
    void parseLine(const QString key, const QVariant val);

    bool prepareForAcquisition(BlackChirp::FtmwType t);
    void setAwgMult(const double m);
    void setUpMixSideband(const BlackChirp::Sideband s);
    void setChirpMult(const double m);
    void setDownMixSideband(const BlackChirp::Sideband s);
    void setCommonLO(bool b);
    void setShotsPerClockStep(int s);
    void setTargetSweeps(int s);
    void setClockDesiredFreq(BlackChirp::ClockType t, double targetFreqMHz);
    void setClockFactor(BlackChirp::ClockType t, double factor);
    void setClockOp(BlackChirp::ClockType t, MultOperation o);
    void setClockOutputNum(BlackChirp::ClockType t, int output);
    void setClockHwKey(BlackChirp::ClockType t, QString key);
    void setClockHwInfo(BlackChirp::ClockType t, QString hwKey, int output);
    void setClockFreqInfo(BlackChirp::ClockType t, double targetFreqMHz = 0.0, double factor = 1.0, MultOperation o = Multiply, QString hwKey = QString(""), int output = 0);
    void setClockFreqInfo(BlackChirp::ClockType t, const ClockFreq &cf);
    void addClockStep(QHash<BlackChirp::ClockType,ClockFreq> h);
    void addClockStep(double upLoMHz, double downLoMHz);
    void clearClockSteps();
    void clearChirpConfigs();
    bool setChirpConfig(const ChirpConfig cc, int num=0);
    void addChirpConfig(ChirpConfig cc);
    int advanceClockStep();


    double awgMult() const;
    BlackChirp::Sideband upMixSideband() const;
    double chirpMult() const;
    BlackChirp::Sideband downMixSideband() const;
    bool commonLO() const;
    int targetSweeps() const;
    int shotsPerClockStep() const;
    int currentIndex() const;
    int completedSweeps() const;
    qint64 totalShots() const;
    qint64 completedSegmentShots() const;
    bool canAdvance(qint64 shots) const;
    int numSegments() const;

    QHash<BlackChirp::ClockType,ClockFreq> getClocks() const;
    double clockFrequency(BlackChirp::ClockType t) const;
    double rawClockFrequency(BlackChirp::ClockType t) const;
    QString clockHardware(BlackChirp::ClockType t) const;
    ChirpConfig getChirpConfig(int num=0) const;
    int numChirpConfigs() const;
    bool isComplete() const;

    double calculateChirpFreq(double awgFreq) const;
    double calculateAwgFreq(double chirpFreq) const;
    double calculateChirpAbsOffset(double awgFreq) const;
    QPair<double,double> calculateChirpAbsOffsetRange() const;

    QString clockStepsString() const;
    void loadClockSteps(int num, QString path);

private:
    QSharedDataPointer<RfConfigData> data;

    double getRawFrequency(ClockFreq f) const;
};

Q_DECLARE_METATYPE(RfConfig)
Q_DECLARE_METATYPE(RfConfig::MultOperation)

#endif // RFCONFIG_H
