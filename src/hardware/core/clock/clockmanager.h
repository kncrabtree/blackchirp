#ifndef CLOCKMANAGER_H
#define CLOCKMANAGER_H

#include <QObject>
#include <QMultiHash>

#include <data/storage/settingsstorage.h>
#include <data/experiment/experiment.h>
#include <data/settings/hardwarekeys.h>

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
    QHash<RfConfig::ClockType,RfConfig::ClockFreq> getCurrentClocks();
    double setClockFrequency(RfConfig::ClockType t, double freqMHz);
    double readClockFrequency(RfConfig::ClockType t);
    bool configureClocks(QHash<RfConfig::ClockType,RfConfig::ClockFreq> clocks);
    bool prepareForExperiment(Experiment &exp);

    // Public API for HardwareManager integration
    QVector<Clock*> getClockList() const;
    void createClocksFromRuntimeConfig();

private:
    QVector<Clock*> d_clockList;
    QHash<RfConfig::ClockType,Clock*> d_clockRoles;
    
    void setupClocks();

};

#endif // CLOCKMANAGER_H
