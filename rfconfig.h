#ifndef RFCONFIG_H
#define RFCONFIG_H

#include <QSharedDataPointer>

#include "datastructs.h"
#include "chirpconfig.h"

class RfConfigData;

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
    };

    RfConfig();
    RfConfig(const RfConfig &);
    RfConfig &operator=(const RfConfig &);
    ~RfConfig();

    void saveToSetting() const;
    static RfConfig loadFromSettings();

    bool isValid() const;
    void setAwgMult(const double m);
    void setUpMixSideband(const BlackChirp::Sideband s);
    void setChirpMult(const double m);
    void setDownMixSideband(const BlackChirp::Sideband s);
    void setCommonLO(bool b);
    void setClockFreq(BlackChirp::ClockType t, double targetFreqMHz, double factor = 1.0, MultOperation o = Multiply);
    void clearChirpConfigs();
    bool setChirpConfig(const ChirpConfig cc, int num=0);
    void addChirpConfig(const ChirpConfig cc);

    double awgMult() const;
    BlackChirp::Sideband upMixSideband() const;
    double chirpMult() const;
    BlackChirp::Sideband downMixSideband() const;
    bool commonLO() const;
    double clockFrequency(BlackChirp::ClockType t) const;
    double rawClockFrequency(BlackChirp::ClockType t) const;
    ChirpConfig getChirpConfig(int num=0);

    double calculateChirpFreq(double awgFreq) const;
    double calculateAwgFreq(double chirpFreq) const;

private:
    QSharedDataPointer<RfConfigData> data;

    double getRawFrequency(ClockFreq f) const;
};

#endif // RFCONFIG_H
