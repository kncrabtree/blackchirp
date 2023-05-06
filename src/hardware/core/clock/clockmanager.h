#ifndef CLOCKMANAGER_H
#define CLOCKMANAGER_H

#include <QObject>
#include <QMultiHash>

#include <data/storage/settingsstorage.h>
#include <data/experiment/experiment.h>
#include <boost/preprocessor/iteration/local.hpp>

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

//handle includes... haven't figure out how to do this in a macro
#include "clock_h.h"

#endif // CLOCKMANAGER_H
