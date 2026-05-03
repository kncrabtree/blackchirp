#ifndef CLOCKMANAGER_H
#define CLOCKMANAGER_H

#include <QObject>
#include <QMultiHash>

#include <data/storage/settingsstorage.h>
#include <data/experiment/experiment.h>
#include <data/settings/hardwarekeys.h>

class Clock;

/// \brief Routes RF clock operations to the correct \c Clock hardware objects by role.
///
/// \c ClockManager owns the role-to-hardware mapping for all clock sources in
/// the active loadout.  Each \c RfConfig::ClockType role (e.g., \c UpLO,
/// \c DownLO, \c AwgRef) is served by exactly one \c Clock object on one of
/// that clock's numbered outputs.  A single physical clock may serve multiple
/// roles on different outputs simultaneously.
///
/// \c ClockManager is created and owned by \c HardwareManager
/// (\c pu_clockManager) and runs on the HardwareManager thread.
/// \c HardwareManager hands off the current \c Clock* pointers via
/// \c setClocksFromHardwareManager() after each hardware sync cycle, and calls
/// \c updateClockManager() whenever the set of available clocks changes.
/// Direct invocation of \c ClockManager slots from other threads is not safe;
/// all cross-thread callers must go through \c HardwareManager's queued
/// connections.
///
/// \sa HardwareManager, RfConfig, Clock
class ClockManager : public QObject, public SettingsStorage
{
    Q_OBJECT
public:
    /// \brief Constructs a \c ClockManager with an empty clock list.
    ///
    /// The manager reads its \c SettingsStorage state from the
    /// \c BC::Key::ClockManager::clockManager key.  Clock objects must be
    /// supplied by \c HardwareManager via \c setClocksFromHardwareManager()
    /// before any configure or read operations are meaningful.
    /// \param parent Optional parent \c QObject.
    explicit ClockManager(QObject *parent = nullptr);

signals:
    /// \brief Emitted when the frequency of a clock role changes.
    ///
    /// Connected by \c setupClocks() to each \c Clock::frequencyUpdate signal
    /// and re-emitted by \c HardwareManager to the rest of the system.
    /// \param type The logical clock role whose frequency has changed.
    /// \param freqMHz The new frequency in MHz.
    void clockFrequencyUpdate(RfConfig::ClockType type, double freqMHz);

    /// \brief Emitted when the hardware binding for a clock role changes.
    ///
    /// Fires once per role when \c configureClocks() assigns or clears a role.
    /// If a role is removed (e.g., the clock map no longer includes it), this
    /// signal is emitted with an empty \a hwKey and \a output equal to \c -1.
    /// \param type The logical clock role that was reassigned.
    /// \param hwKey Hardware key of the newly assigned \c Clock, or an empty
    ///        string if the role is now unassigned.
    /// \param output Output port index on the \c Clock device, or \c -1 if
    ///        the role is unassigned.
    void clockHardwareUpdate(RfConfig::ClockType type, const QString &hwKey, int output);

public slots:
    /// \brief Reads the current hardware frequency for every active clock role.
    ///
    /// Iterates over \c d_clockRoles and calls \c Clock::readFrequency() for
    /// each role whose clock reports \c isConnected().  Each successful read
    /// causes the corresponding \c Clock to emit \c frequencyUpdate, which
    /// propagates through \c clockFrequencyUpdate.
    ///
    /// Must be called on the HardwareManager thread.
    void readActiveClocks();

    /// \brief Returns the current frequency descriptor for every active clock role.
    ///
    /// Queries each entry in \c d_clockRoles for its hardware key, output
    /// index, multiplication factor, and live frequency, and assembles the
    /// results into a hash that can be written back into an \c RfConfig via
    /// \c RfConfig::setCurrentClocks().
    ///
    /// Must be called on the HardwareManager thread.
    /// \return A hash mapping each active \c RfConfig::ClockType to its
    ///         \c RfConfig::ClockFreq descriptor.
    QHash<RfConfig::ClockType,RfConfig::ClockFreq> getCurrentClocks();

    /// \brief Sets the frequency of the clock assigned to \a t.
    ///
    /// Forwards the call to \c Clock::setFrequency() on the \c Clock* in
    /// \c d_clockRoles for \a t.  Logs a warning and returns \c -1.0 if no
    /// clock is currently assigned to \a t.
    ///
    /// Must be called on the HardwareManager thread.
    /// \param t The clock role to tune.
    /// \param freqMHz Desired output frequency in MHz.
    /// \return Actual achieved frequency in MHz, or \c -1.0 on failure.
    double setClockFrequency(RfConfig::ClockType t, double freqMHz);

    /// \brief Reads the current frequency of the clock assigned to \a t.
    ///
    /// Forwards the call to \c Clock::readFrequency() on the assigned
    /// \c Clock*.  Logs a warning and returns \c -1.0 if no clock is assigned
    /// to \a t.
    ///
    /// Must be called on the HardwareManager thread.
    /// \param t The clock role to query.
    /// \return Current frequency in MHz, or \c -1.0 on failure.
    double readClockFrequency(RfConfig::ClockType t);

    /// \brief Applies a complete clock configuration to hardware.
    ///
    /// Clears all existing role assignments, then for each entry in \a clocks:
    /// resolves the hardware key to a \c Clock* in \c d_clockList, registers
    /// the role on the correct output via \c Clock::addRole(), writes the
    /// multiplication factor via \c Clock::setMultFactor(), calls
    /// \c Clock::setFrequency(), and emits \c clockHardwareUpdate().  Roles
    /// present before the call but absent from \a clocks receive a
    /// \c clockHardwareUpdate with an empty key and output \c -1 to signal
    /// that they are no longer active.
    ///
    /// Returns \c false (and logs an error) if a hardware key in \a clocks
    /// cannot be matched to a known \c Clock, if the requested output index is
    /// out of range, or if a \c setFrequency call fails.
    ///
    /// Must be called on the HardwareManager thread.
    /// \param clocks Desired role-to-frequency mapping as produced by
    ///        \c RfConfig::getClocks().
    /// \return \c true if all roles were configured successfully;
    ///         \c false on the first error.
    bool configureClocks(QHash<RfConfig::ClockType,RfConfig::ClockFreq> clocks);

    /// \brief Configures clocks for an experiment and writes achieved
    ///        frequencies back into the experiment object.
    ///
    /// Returns \c true immediately if the experiment does not have FTMW
    /// enabled.  Otherwise calls \c configureClocks() with the clock map from
    /// \c exp.ftmwConfig()->d_rfConfig.getClocks(), and on success calls
    /// \c RfConfig::setCurrentClocks(getCurrentClocks()) to record the
    /// achieved frequencies in \a exp.
    ///
    /// Must be called on the HardwareManager thread.
    /// \param exp The experiment to prepare.  Modified in place with the
    ///        actual clock frequencies after configuration.
    /// \return \c true on success or if FTMW is disabled; \c false if
    ///         \c configureClocks() fails.
    bool prepareForExperiment(Experiment &exp);

    /// \brief Accepts a new set of \c Clock pointers from \c HardwareManager.
    ///
    /// Replaces \c d_clockList with \a clocks (which remain owned by
    /// \c HardwareManager) and calls \c setupClocks() to rebuild the role map
    /// and re-connect \c frequencyUpdate signals.
    ///
    /// Must be called on the HardwareManager thread.
    /// \param clocks Ordered list of \c Clock* objects currently in the
    ///        hardware map.
    void setClocksFromHardwareManager(const QVector<Clock*>& clocks);

    /// \brief Rebuilds the internal clock setup from the current \c d_clockList.
    ///
    /// Called by \c HardwareManager when the hardware map changes after
    /// \c setClocksFromHardwareManager() has already been called.  Equivalent
    /// to re-running \c setupClocks() on the existing pointer list.
    ///
    /// Must be called on the HardwareManager thread.
    void reconfigureFromRuntimeConfig();

    /// \brief Returns the current list of \c Clock pointers held by this manager.
    ///
    /// \return A copy of \c d_clockList as supplied by the most recent
    ///         \c setClocksFromHardwareManager() call.
    QVector<Clock*> getClockList() const;

private:
    /// \brief Ordered list of all \c Clock objects in the active loadout.
    ///
    /// Pointers are borrowed from \c HardwareManager; lifetime is managed by
    /// \c HardwareManager.  Updated by \c setClocksFromHardwareManager().
    QVector<Clock*> d_clockList;

    /// \brief Maps each active \c RfConfig::ClockType role to the \c Clock* that serves it.
    ///
    /// Rebuilt by \c configureClocks().  A role is present in this hash only
    /// when a clock has been successfully assigned to it.  Multiple roles may
    /// map to the same \c Clock* (each on a different output index).
    QHash<RfConfig::ClockType,Clock*> d_clockRoles;

    /// \brief Populates the \c SettingsStorage output table and re-connects signals.
    ///
    /// Iterates \c d_clockList, writes each clock's outputs to the
    /// \c BC::Key::ClockManager::hwClocks array in \c SettingsStorage, emits
    /// \c clockHardwareUpdate for roles that are already assigned on each
    /// clock, and connects \c Clock::frequencyUpdate to
    /// \c ClockManager::clockFrequencyUpdate.  Called from
    /// \c setClocksFromHardwareManager() and \c reconfigureFromRuntimeConfig().
    void setupClocks();

};

#endif // CLOCKMANAGER_H
