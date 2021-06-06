#ifndef CLOCK_H
#define CLOCK_H

#include <src/hardware/core/hardwareobject.h>

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
    explicit Clock(int clockNum, QObject *parent = nullptr);

    int numOutputs() { return d_numOutputs; }
    bool isTunable() { return d_isTunable; }
    void setMultFactor(double d, int output=0);

    //implement this function if d_numChannels > 1 to return
    //human-readable names for each output (e.g., Source 1, Source 2, etc)
    virtual QStringList channelNames();

public slots:
    void initialize() override final;
    bool addRole(BlackChirp::ClockType t, int outputIndex = 0);
    void removeRole(BlackChirp::ClockType t);
    void clearRoles();
    bool hasRole(BlackChirp::ClockType t);

    void readAll();
    double readFrequency(BlackChirp::ClockType t);
    double setFrequency(BlackChirp::ClockType t, double freqMHz);

signals:
    void frequencyUpdate(BlackChirp::ClockType, double);


protected:
    int d_numOutputs;
    bool d_isTunable;
    double d_minFreqMHz, d_maxFreqMHz;
    QHash<BlackChirp::ClockType,int> d_outputRoles;
    QList<double> d_multFactors;

    virtual void initializeClock() =0;
    virtual bool setHwFrequency(double freqMHz, int outputIndex = 0) =0;
    virtual double readHwFrequency(int outputIndex = 0) =0;
    virtual bool prepareClock(Experiment &exp) { return true; }

    // HardwareObject interface
public slots:
    virtual bool prepareForExperiment(Experiment &exp) override final;
};

#endif // CLOCK_H
