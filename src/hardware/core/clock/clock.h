#ifndef CLOCK_H
#define CLOCK_H

#include <hardware/core/hardwareobject.h>

namespace BC::Key::Clock {
static const QString minFreq("minFreqMHz");
static const QString maxFreq("maxFreqMHz");
static const QString lock("lockExternal");
static const QString outputs("outputs");
static const QString mf("multFactor");
static const QString role("role");
}

/**
 * @brief The Clock class defines an interface for an oscillator
 *
 * Unlike most other HardwareObjects, BlackChirp expects to have multiple clocks.
 * As a result, the constructor of the clock needs to receive information about
 * which clock this is for the purpose of identifying it in QSettings.
 *
 * Also unlike most other HardwareObjects, the role of a particular clock can be
 * configured at runtime. This allows for a single hardware implementation (e.g.,
 * the Valon5009) to be used for different purposes across instruments without
 * needing to write separate code.
 *
 * The Clock class keeps track of which role(s) it serves; 1 role per output.
 * It also tells the ClockManager whether it is tunable and what its allowed
 * frequency range is (though the range itself is defined in its implementation).
 *
 *
 *
 */
class Clock : public HardwareObject
{
    Q_OBJECT
public:
    explicit Clock(int clockNum, int numOutputs, bool tunable, const QString subKey, const QString name,
                   CommunicationProtocol::CommType commType, QObject *parent = nullptr);
    virtual ~Clock();

    int numOutputs() { return d_numOutputs; }
    bool isTunable() { return d_isTunable; }
    void setMultFactor(double d, int output=0);
    double multFactor(int output=0);
    int outputForRole(RfConfig::ClockType t);

    //implement this function if d_numChannels > 1 to return
    //human-readable names for each output (e.g., Source 1, Source 2, etc)
    virtual QStringList channelNames();

public slots:
    void initialize() override final;
    bool testConnection() override final;
    bool addRole(RfConfig::ClockType t, int outputIndex = 0);
    void removeRole(RfConfig::ClockType t);
    void clearRoles();
    bool hasRole(RfConfig::ClockType t);

    void readAll();
    double readFrequency(RfConfig::ClockType t);
    double setFrequency(RfConfig::ClockType t, double freqMHz);

signals:
    void frequencyUpdate(RfConfig::ClockType, double);

private:
    int d_numOutputs;
    bool d_isTunable;
    QHash<RfConfig::ClockType,int> d_outputRoles;
    QVector<double> d_multFactors;


protected:
    virtual void initializeClock() =0;
    virtual bool testClockConnection() =0;
    virtual bool setHwFrequency(double freqMHz, int outputIndex = 0) =0;
    virtual double readHwFrequency(int outputIndex = 0) =0;
    virtual bool prepareClock(Experiment &exp) { Q_UNUSED(exp) return true; }

    // HardwareObject interface
public slots:
    virtual bool prepareForExperiment(Experiment &exp) override final;
};

#endif // CLOCK_H
