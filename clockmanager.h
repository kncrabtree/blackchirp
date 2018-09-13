#ifndef CLOCKMANAGER_H
#define CLOCKMANAGER_H

#include <QObject>

#include "datastructs.h"
#include "experiment.h"

class Clock;

/**
 * @brief The ClockManager class associates hardware clocks with their purposes
 *
 * CP-FTMW experiments use a variety of clock frequencies. In simple cases, there
 * may be a PLDRO that is used to mix the chirp up and/or down in frequency, but in
 * more complex cases mutlitple clocks can be used to perform upconversion, downconversion,
 * AWG sample clocking, digitizer clocking, or even more.
 *
 * The ClockManager class is intended to figure out which hardware clock serves each
 * purpose, and route read/write commands to the appropriate sources.
 * Some clock objects (e.g., Valon 5009) have multiple independent outputs, so the
 * ClockManager may associate multiple functions with a single hardware object.
 *
 */

class ClockManager : public QObject
{
    Q_OBJECT
public:
    explicit ClockManager(QObject *parent = nullptr);
    QList<Clock*> clockList() { return d_clockList; }

signals:
    void logMessage(QString, BlackChirp::LogMessageCode mc = BlackChirp::LogNormal);
    void clockFrequencyUpdate(BlackChirp::ClockType, double);

public slots:
    double setClockFrequency(BlackChirp::ClockType t, double freqMHz);
    double readClockFrequency(BlackChirp::ClockType t);
    Experiment prepareForExperiment(Experiment exp);

private:
    QList<Clock*> d_clockList;
    QList<BlackChirp::ClockType> d_clockTypes;
    QMap<BlackChirp::ClockType,Clock*> d_clockRoles;

};

#endif // CLOCKMANAGER_H

//Define clock types for each clock


#ifdef BC_CLOCK_FIXED
#include "fixedclock.h"
class FixedClock;
#endif

#ifdef BC_CLOCK_0
typedef BC_CLOCK_0 Clock0Hardware;
#endif

#ifdef BC_CLOCK_1
typedef BC_CLOCK_1 Clock1Hardware;
#endif
