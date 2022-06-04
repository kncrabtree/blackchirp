#ifndef CLOCKMANAGER_H
#define CLOCKMANAGER_H

#include <QObject>
#include <QMultiHash>

#include <data/storage/settingsstorage.h>
#include <data/experiment/experiment.h>

class Clock;

namespace BC::Key::Clock {
static const QString clockManager{"ClockManager"};
static const QString hwClocks{"hwClocks"};
static const QString clockKey{"key"};
static const QString clockOutput{"output"};
static const QString clockName{"name"};
}

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
class ClockManager : public QObject, public SettingsStorage
{
    Q_OBJECT
public:
    explicit ClockManager(QObject *parent = nullptr);

signals:
    void logMessage(QString, LogHandler::MessageCode mc = LogHandler::Normal);
    void clockFrequencyUpdate(RfConfig::ClockType, double);

public slots:
    void readActiveClocks();
    QMultiHash<RfConfig::ClockType,RfConfig::ClockFreq> getCurrentClocks();
    double setClockFrequency(RfConfig::ClockType t, double freqMHz);
    double readClockFrequency(RfConfig::ClockType t);
    bool configureClocks(QMultiHash<RfConfig::ClockType,RfConfig::ClockFreq> clocks);
    bool prepareForExperiment(Experiment &exp);

private:
    QVector<Clock*> d_clockList;
    QMultiHash<RfConfig::ClockType,Clock*> d_clockRoles;

    friend class HardwareManager;

};

#endif // CLOCKMANAGER_H

//Define clock types for each clock


#ifdef BC_CLOCK_FIXED
#include "fixedclock.h"
class FixedClock;
#endif

#ifdef BC_CLOCK_VALON5009
#include "valon5009.h"
class Valon5009;
#endif

#ifdef BC_CLOCK_VALON5015
#include "valon5015.h"
class Valon5015;
#endif

#ifdef BC_CLOCK_0
using Clock0Hardware = BC_CLOCK_0;
#endif

#ifdef BC_CLOCK_1
using Clock1Hardware = BC_CLOCK_1;
#endif

#ifdef BC_CLOCK_2
using Clock2Hardware = BC_CLOCK_2;
#endif

#ifdef BC_CLOCK_3
using Clock3Hardware = BC_CLOCK_3;
#endif

#ifdef BC_CLOCK_4
using Clock4Hardware = BC_CLOCK_4;
#endif
