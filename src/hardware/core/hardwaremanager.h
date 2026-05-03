#ifndef HARDWAREMANAGER_H
#define HARDWAREMANAGER_H

#include <QObject>
#include <memory>
#include <functional>
#include <atomic>
#include <data/loghandler.h>
#include <data/storage/auxdatastorage.h>
#include <data/storage/settingsstorage.h>
#include <data/experiment/rfconfig.h>
#include <QMutex>
#include <QReadWriteLock>

#include <data/experiment/hardware/optional/flowcontroller/flowconfig.h>
#include <data/experiment/hardware/optional/pulsegenerator/pulsegenconfig.h>
#include <data/experiment/hardware/optional/pressurecontroller/pressurecontrollerconfig.h>
#include <data/experiment/hardware/optional/tempcontroller/temperaturecontrollerconfig.h>
#include <data/experiment/hardware/optional/ioboard/ioboardconfig.h>

#include <data/lif/lifconfig.h>
#include <hardware/core/communication/communicationprotocol.h>

class HardwareObject;
class ClockManager;
class Experiment;
class GpibController;

namespace BC::Key {
inline constexpr QLatin1StringView hw{"hardware"};
inline constexpr QLatin1StringView allHw{"instruments"};
}

/// \brief Orchestration layer that owns all live HardwareObject instances, marshals
/// cross-thread calls, and fans out connection and data notifications to the GUI and
/// acquisition subsystem.
///
/// HardwareManager lives on a dedicated QThread created by MainWindow. Every
/// HardwareObject it manages may live on its own further thread; HardwareManager
/// mediates all communication via Qt's queued-connection and
/// QMetaObject::invokeMethod mechanisms so that callers on any thread can invoke
/// hardware operations safely.
///
/// The manager's public slot surface is the primary interface through which the GUI
/// and AcquisitionManager request hardware operations. Its public signal surface is
/// the source of all hardware-state notifications: connection results, sensor data,
/// clock frequency updates, and experiment lifecycle events.
///
/// Hardware is loaded from RuntimeHardwareConfig on startup via initialize() and can
/// be changed at runtime via syncWithRuntimeConfig() and applyHardwareMap(). The
/// internal map d_hardwareMap is protected by d_hardwareMapLock; connection-test
/// state is protected by the separate d_connectionStateLock to avoid holding the
/// map lock during potentially long connection operations.
///
/// \sa HardwareObject, RuntimeHardwareConfig, LoadoutManager, ClockManager,
///     Experiment
class HardwareManager : public QObject, public SettingsStorage
{
    Q_OBJECT
public:
    /// \brief Constructs HardwareManager and wires ClockManager; hardware
    /// objects are not created until initialize() is called.
    explicit HardwareManager(QObject *parent = 0);

    /// \brief Destroys the manager, quitting and joining any per-hardware threads.
    ~HardwareManager();

    /// \brief Returns the single live instance for const (read-only) hardware
    /// resolution from any thread.
    ///
    /// This static accessor allows code that cannot hold a reference to the
    /// manager (e.g., hardware objects themselves resolving GPIB controllers) to
    /// reach the map without acquiring any write-capable handle. Callers must not
    /// store the reference beyond the current call stack; the instance is valid
    /// only for the application lifetime.
    ///
    /// \return Const reference to the HardwareManager singleton.
    /// \warning Throws std::runtime_error if called before the first
    ///          HardwareManager is constructed.
    static const HardwareManager& constInstance();

    /// \brief Returns the display name string for the hardware identified by \a key.
    /// \param key Hardware key of the form \c "<Type>.<label>".
    /// \return The hardware object's d_key field, or an empty string if not found.
    QString getHwName(const QString key);

    /// \brief Set of optional hardware type names recognized by storeAllOptHw().
    ///
    /// Contains the class names (from \c staticMetaObject.className()) of each
    /// optional hardware category: FlowController, IOBoard, PressureController,
    /// PulseGenerator, and TemperatureController.
    std::set<QString> d_optHwTypes;

signals:
    /// \brief Emitted to display a transient message in the status bar.
    /// \param msg The message text.
    /// \param timeout Display duration in milliseconds; 0 means indefinite.
    void statusMessage(QString,int=0);

    /// \brief Emitted when all hardware objects have been constructed, threaded,
    /// and their initialization slots invoked — before any connection tests run.
    void hwInitializationComplete();

    /// \brief Emitted after every connection-test round to report whether all
    /// critical hardware responded successfully.
    /// \param success \c true when every critical HardwareObject is connected.
    void allHardwareConnected(bool);

    /*!
     * \brief Unified signal for all connection status changes and test results.
     *
     * Emitted by handleConnectionResult() when a hardware object reports a
     * connection attempt outcome, by hardwareFailure() when a previously
     * connected object signals failure, and by removeHardwareInternal() when an
     * object is removed from the map. Consumers can use this single signal to
     * maintain a current connection-status view across the entire hardware set.
     *
     * \param hwKey Hardware key (e.g., \c "FtmwScope.mainScope").
     * \param success \c true if the connection is now active.
     * \param msg Human-readable status or error message; may be empty.
     */
    void connectionResult(const QString& hwKey, bool success, const QString& msg);

    /// \brief Emitted when a hardware profile is permanently deleted (not just
    /// deactivated) and its settings have been purged.
    /// \param hwKey The hardware key whose profile was removed.
    void profileDeleted(const QString& hwKey);

    // Communication protocol management signals

    /// \brief Emitted in response to getHardwareCommunicationInfo() with the
    /// current and supported communication protocols for a device.
    /// \param hwKey Hardware key of the queried device.
    /// \param currentProtocol The protocol currently in use.
    /// \param supportedProtocols All protocols the device implementation supports.
    /// \param connected Whether the device is currently connected.
    void hardwareCommunicationInfoReady(const QString& hwKey, CommunicationProtocol::CommType currentProtocol,
                                       QVector<CommunicationProtocol::CommType> supportedProtocols, bool connected);

    /// \brief Emitted after setHardwareProtocol() completes to report whether
    /// the protocol change succeeded.
    /// \param hwKey Hardware key of the affected device.
    /// \param success \c true if the protocol was changed and reconnected.
    /// \param msg Diagnostic message; empty on success.
    void protocolSetResult(const QString& hwKey, bool success, const QString& msg);

    /// \brief Emitted by getActiveGpibControllers() with the keys of all
    /// GpibController objects currently in the hardware map.
    /// \param controllerKeys List of hardware keys for available GPIB controllers.
    void gpibControllersAvailable(QStringList controllerKeys);

    // Acquisition lifecycle signals

    /// \brief Broadcast to all HardwareObjects to enter acquisition mode.
    void beginAcquisition();

    /// \brief Broadcast to all HardwareObjects to abort the current acquisition.
    void abortAcquisition();

    /// \brief Emitted after initializeExperiment() has prepared all hardware and
    /// the Experiment object is ready for the acquisition loop to start.
    /// \param exp The prepared experiment, shared with AcquisitionManager.
    void experimentInitialized(std::shared_ptr<Experiment>);

    /// \brief Broadcast to all HardwareObjects to exit acquisition mode.
    void endAcquisition();

    // Auxiliary data signals

    /// \brief Emitted when any hardware object reads auxiliary (non-rolling)
    /// data; keys are prefixed with the hardware key of the source object.
    /// \param data Map of keyed scalar values.
    void auxData(AuxDataStorage::AuxDataMap);

    /// \brief Emitted with the same data as auxData() for validation tracking.
    /// \param data Map of keyed scalar values.
    void validationData(AuxDataStorage::AuxDataMap);

    /// \brief Emitted when any hardware object reads rolling (time-series)
    /// auxiliary data; keys are prefixed with the hardware key of the source.
    /// \param data Map of keyed scalar values.
    /// \param timestamp Acquisition timestamp for the data point.
    void rollingData(AuxDataStorage::AuxDataMap,QDateTime);

    // Clock signals

    /// \brief Emitted when a single clock frequency changes during an
    /// experiment (e.g., a sweep step).
    /// \param type The clock role that changed.
    /// \param freqMHz New frequency in MHz.
    void clockFrequencyUpdate(RfConfig::ClockType, double);

    /// \brief Emitted when the hardware assignment for a clock role changes.
    /// \param type The clock role.
    /// \param hwKey Key of the clock hardware now serving this role.
    /// \param output Hardware output index used for this role.
    void clockHardwareUpdate(RfConfig::ClockType, const QString &hwKey, int output);

    /// \brief Emitted by setClocks() after all requested clock frequencies have
    /// been applied and the digitizer ungated.
    /// \param clocks Map of clock roles to their final frequency descriptors.
    void allClocksReady(QHash<RfConfig::ClockType,RfConfig::ClockFreq>);

    // Pulse generator signals

    /// \brief Emitted when a single pulse-generator channel setting changes.
    /// \param hwKey Hardware key of the pulse generator.
    /// \param channel Channel index.
    /// \param setting Which setting changed.
    /// \param value New value.
    void pGenSettingUpdate(QString,int,PulseGenConfig::Setting,QVariant);

    /// \brief Emitted when the entire pulse generator configuration is updated.
    /// \param hwKey Hardware key of the pulse generator.
    /// \param config New full configuration.
    void pGenConfigUpdate(QString,PulseGenConfig);

    // Flow controller signals

    /// \brief Emitted when a flow-controller channel flow reading changes.
    /// \param hwKey Hardware key of the flow controller.
    /// \param channel Channel index.
    /// \param value Flow reading in the controller's native units.
    void flowUpdate(QString,int,double);

    /// \brief Emitted when a flow-controller channel setpoint changes.
    /// \param hwKey Hardware key of the flow controller.
    /// \param channel Channel index.
    /// \param value New setpoint in the controller's native units.
    void flowSetpointUpdate(QString,int,double);

    /// \brief Emitted when the backing-gas pressure reading from a flow
    /// controller changes.
    /// \param hwKey Hardware key of the flow controller.
    /// \param value Pressure reading.
    void gasPressureUpdate(QString,double);

    /// \brief Emitted when the backing-gas pressure setpoint changes.
    /// \param hwKey Hardware key of the flow controller.
    /// \param value New setpoint.
    void gasPressureSetpointUpdate(QString,double);

    /// \brief Emitted when the backing-gas pressure control mode changes.
    /// \param hwKey Hardware key of the flow controller.
    /// \param enabled \c true if pressure feedback control is active.
    void gasPressureControlMode(QString,bool);

    // Pressure controller signals

    /// \brief Emitted when the pressure controller becomes read-only (e.g.,
    /// during an experiment).
    /// \param hwKey Hardware key of the pressure controller.
    /// \param readOnly \c true if write operations are disabled.
    void pressureControlReadOnly(QString,bool);

    /// \brief Emitted when the pressure reading from a dedicated pressure
    /// controller changes.
    /// \param hwKey Hardware key of the pressure controller.
    /// \param value Pressure reading.
    void pressureUpdate(QString,double);

    /// \brief Emitted when the pressure setpoint changes.
    /// \param hwKey Hardware key of the pressure controller.
    /// \param value New setpoint.
    void pressureSetpointUpdate(QString,double);

    /// \brief Emitted when the pressure control mode changes.
    /// \param hwKey Hardware key of the pressure controller.
    /// \param enabled \c true if feedback control is active.
    void pressureControlMode(QString,bool);

    // Temperature controller signals

    /// \brief Emitted when a temperature channel reading changes.
    /// \param hwKey Hardware key of the temperature controller.
    /// \param channel Channel index.
    /// \param value Temperature reading.
    void temperatureUpdate(QString,uint,double);

    /// \brief Emitted when a temperature channel is enabled or disabled.
    /// \param hwKey Hardware key of the temperature controller.
    /// \param channel Channel index.
    /// \param enabled \c true if the channel is active.
    void temperatureEnableUpdate(QString,uint,bool);

    // LIF signals

    /// \brief Emitted when the LIF digitizer acquires a single shot waveform
    /// during configuration acquisition.
    /// \param data Raw 8-bit waveform samples.
    void lifScopeShotAcquired(QVector<qint8>);

    /// \brief Emitted when LIF hardware setup (laser position, pulse delay,
    /// digitizer gate) is complete.
    /// \param success \c true if all LIF hardware accepted the requested parameters.
    void lifSettingsComplete(bool success = true);

    /// \brief Emitted when the LIF laser reports a new position.
    /// \param pos Laser position in the controller's native units.
    void lifLaserPosUpdate(double);

    /// \brief Emitted when the LIF digitizer completes configuration acquisition
    /// and the system is ready to begin data collection.
    void lifConfigAcqStarted();

    /// \brief Emitted when the LIF laser flashlamp enable state changes.
    /// \param enabled \c true if the flashlamp is active.
    void lifLaserFlashlampUpdate(bool);

    // Python hardware signal

    /// \brief Emitted after reloadPythonScript() completes for a Python-backed
    /// hardware object.
    /// \param hwKey Hardware key of the reloaded device.
    /// \param success \c true if the script was reloaded and the object
    ///        re-initialized successfully.
    /// \param msg Diagnostic message; empty on success.
    void pythonScriptReloadResult(const QString &hwKey, bool success, const QString &msg);

public slots:
    /// \brief Loads hardware from RuntimeHardwareConfig, starts per-hardware
    /// threads, and emits hwInitializationComplete() when the map is populated.
    ///
    /// Called once by the owning QThread's started() signal. Ensures system
    /// profiles exist, calls syncWithRuntimeConfig() to populate d_hardwareMap,
    /// and logs warnings for virtual instruments before emitting the completion
    /// signal.
    void initialize();

    /*!
     * \brief Records whether hardware connection was successful.
     *
     * Invoked (typically via a queued connection) when a HardwareObject
     * emits its \c connected signal. Updates d_connectionState, emits
     * connectionResult(), and calls checkStatus() to determine whether
     * allHardwareConnected() should fire.
     *
     * \param hwKey Hardware key of the tested object.
     * \param success Whether communication was successful.
     * \param msg Error message; empty on success.
     */
    void handleConnectionResult(const QString& hwKey, bool success, const QString& msg);

    /*!
     * \brief Handles a hardware failure signal from a previously connected device.
     *
     * Disconnects the hardwareFailure connection so the signal is not
     * re-processed, emits connectionResult() with \c success=false, and — if
     * the failing device is critical — emits abortAcquisition() to terminate
     * any in-progress experiment.
     *
     * \note The sender() is expected to be the HardwareObject that failed.
     * \warning Consider generating an abort signal here for non-critical failures.
     */
    void hardwareFailure();

    /// \brief Puts all connected hardware objects to sleep or wakes them.
    /// \param b \c true to sleep, \c false to wake.
    void sleep(bool b);

    /// \brief Prepares all hardware for the given experiment and emits
    /// experimentInitialized() when complete.
    ///
    /// Configures clocks via ClockManager, calls hwPrepareForExperiment() on
    /// each HardwareObject (blocking across thread boundaries), and validates
    /// LIF hardware availability if the experiment requires it.
    ///
    /// \param exp The experiment to initialize; d_hardwareSuccess is set on
    ///        the object before the signal is emitted.
    void initializeExperiment(std::shared_ptr<Experiment> exp);

    /// \brief Called when an experiment completes; reserved for post-experiment
    /// hardware teardown.
    void experimentComplete();

    /// \brief Triggers a connection test on every hardware object in the map.
    void testAll();

    /// \brief Triggers a connection test on the hardware object identified by
    /// \a hwKey.
    /// \param hwKey Hardware key to test.
    void testObjectConnection(const QString hwKey);

    /// \brief Asks the hardware object identified by \a key to re-read its
    /// settings from QSettings.
    /// \param key Hardware key of the object to update.
    void updateObjectSettings(const QString key);

    /// \brief Polls all hardware objects for auxiliary data and accumulates
    /// the results into the auxData() signal.
    void getAuxData();

    /// \brief Returns the current clock frequencies from ClockManager.
    /// \return Map of clock roles to their current frequency descriptors.
    QHash<RfConfig::ClockType,RfConfig::ClockFreq> getClocks();

    /// \brief Passes a desired clock configuration to ClockManager without
    /// applying it to hardware.
    /// \param clocks Desired clock map.
    void configureClocks(QHash<RfConfig::ClockType,RfConfig::ClockFreq> clocks);

    /// \brief Applies a clock configuration to hardware, gating the FTMW
    /// digitizer during the transition, and emits allClocksReady() when done.
    /// \param clocks Clock map with desired frequencies; actual achieved
    ///        frequencies are written back into the map.
    void setClocks(QHash<RfConfig::ClockType,RfConfig::ClockFreq> clocks);

    /// \brief Sets a single channel setting on the pulse generator identified
    /// by \a key.
    /// \param key Hardware key of the pulse generator.
    /// \param index Channel index.
    /// \param s Which setting to change.
    /// \param val New value.
    void setPGenSetting(const QString key, int index, PulseGenConfig::Setting s, QVariant val);

    /// \brief Applies a complete PulseGenConfig to the pulse generator
    /// identified by \a key.
    /// \param key Hardware key of the pulse generator.
    /// \param c New configuration to apply.
    void setPGenConfig(const QString key, const PulseGenConfig &c);

    /// \brief Returns the current PulseGenConfig from the pulse generator
    /// identified by \a key.
    /// \param key Hardware key of the pulse generator.
    /// \return Current configuration; a default-constructed config if not found.
    PulseGenConfig getPGenConfig(const QString key);

    /// \brief Sets a flow-controller channel setpoint.
    /// \param key Hardware key of the flow controller.
    /// \param index Channel index.
    /// \param val New setpoint.
    void setFlowSetpoint(const QString key, int index, double val);

    /// \brief Renames a flow-controller channel.
    /// \param key Hardware key of the flow controller.
    /// \param index Channel index.
    /// \param name New channel name.
    void setFlowChannelName(const QString key, int index, QString name);

    /// \brief Sets the backing-gas pressure setpoint on a flow controller.
    /// \param key Hardware key of the flow controller.
    /// \param val New setpoint.
    void setGasPressureSetpoint(const QString key, double val);

    /// \brief Enables or disables pressure feedback control on a flow controller.
    /// \param key Hardware key of the flow controller.
    /// \param en \c true to enable control.
    void setGasPressureControlMode(const QString key, bool en);

    /// \brief Returns the current FlowConfig from the flow controller identified
    /// by \a key.
    /// \param key Hardware key of the flow controller.
    /// \return Current configuration; a default-constructed config if not found.
    FlowConfig getFlowConfig(const QString key);

    /// \brief Sets the pressure setpoint on a pressure controller.
    /// \param key Hardware key of the pressure controller.
    /// \param val New setpoint.
    void setPressureSetpoint(const QString key, double val);

    /// \brief Enables or disables feedback control on a pressure controller.
    /// \param key Hardware key of the pressure controller.
    /// \param en \c true to enable control.
    void setPressureControlMode(const QString key, bool en);

    /// \brief Commands the gate valve on a pressure controller to open.
    /// \param key Hardware key of the pressure controller.
    void openGateValve(const QString key);

    /// \brief Commands the gate valve on a pressure controller to close.
    /// \param key Hardware key of the pressure controller.
    void closeGateValve(const QString key);

    /// \brief Returns the current PressureControllerConfig from the pressure
    /// controller identified by \a key.
    /// \param key Hardware key of the pressure controller.
    /// \return Current configuration; a default-constructed config if not found.
    PressureControllerConfig getPressureControllerConfig(const QString key);

    /// \brief Enables or disables a temperature controller channel.
    /// \param key Hardware key of the temperature controller.
    /// \param ch Channel index.
    /// \param en \c true to enable the channel.
    void setTemperatureChannelEnabled(const QString key, uint ch, bool en);

    /// \brief Renames a temperature controller channel.
    /// \param key Hardware key of the temperature controller.
    /// \param ch Channel index.
    /// \param name New channel name.
    void setTemperatureChannelName(const QString key, uint ch, const QString name);

    /// \brief Returns the current TemperatureControllerConfig from the
    /// controller identified by \a key.
    /// \param key Hardware key of the temperature controller.
    /// \return Current configuration; a default-constructed config if not found.
    TemperatureControllerConfig getTemperatureControllerConfig(const QString key);

    /// \brief Returns the current IOBoardConfig from the I/O board identified
    /// by \a key.
    /// \param key Hardware key of the I/O board.
    /// \return Current configuration; a default-constructed config if not found.
    IOBoardConfig getIOBoardConfig(const QString key);

    /// \brief Reads the current state of all optional hardware types and stores
    /// the configs into \a exp.
    ///
    /// Optional hardware types are those in d_optHwTypes. The \a hw map can
    /// selectively suppress reading individual keys by mapping them to \c false.
    ///
    /// \param exp Experiment to populate; configs are appended via addOptHwConfig().
    /// \param hw Optional override map; keys absent from the map are read normally.
    void storeAllOptHw(Experiment *exp, std::map<QString,bool,std::less<>> hw = {});

    // LIF control slots

    /// \brief Sets LIF laser position and pulse delay atomically, gating the LIF
    /// digitizer during the transition, and emits lifSettingsComplete().
    /// \param delay Desired pulse delay for the LIF channel.
    /// \param pos Desired laser position.
    void setLifParameters(double delay, double pos);

    /// \brief Sets the LIF pulse delay on all active pulse generators.
    /// \param d Desired delay.
    /// \return \c true if all pulse generators accepted the new delay.
    bool setPGenLifDelay(double d);

    /// \brief Moves the LIF laser to the requested position.
    /// \param pos Target position.
    /// \return \c true if the laser reported a non-negative achieved position.
    bool setLifLaserPos(double pos);

    /// \brief Starts configuration-mode acquisition on the LIF digitizer.
    /// \param c LIF configuration describing the acquisition parameters.
    void startLifConfigAcq(const LifConfig &c);

    /// \brief Stops configuration-mode acquisition on the LIF digitizer.
    void stopLifConfigAcq();

    /// \brief Returns the current LIF laser position.
    /// \return Position in native units, or \c -1.0 if no laser is available.
    double lifLaserPos();

    /// \brief Returns whether the LIF laser flashlamp is enabled.
    /// \return \c true if the flashlamp is active; \c false if unavailable.
    bool lifLaserFlashlampEnabled();

    /// \brief Enables or disables the LIF laser flashlamp.
    /// \param en \c true to enable the flashlamp.
    void setLifLaserFlashlampEnabled(bool en);

    // Communication protocol management

    /// \brief Queries the communication protocol state of the hardware identified
    /// by \a hwKey and emits hardwareCommunicationInfoReady() with the result.
    /// \param hwKey Hardware key of the device to query.
    void getHardwareCommunicationInfo(const QString& hwKey);

    /// \brief Changes the communication protocol for the hardware identified by
    /// \a hwKey and emits protocolSetResult() when complete.
    /// \param hwKey Hardware key of the target device.
    /// \param protocol The protocol to switch to.
    /// \param gpibControllerKey Key of the GPIB controller to use; ignored
    ///        unless \a protocol is GPIB.
    void setHardwareProtocol(const QString& hwKey, CommunicationProtocol::CommType protocol, const QString& gpibControllerKey = QString());

    /// \brief Enumerates all active GpibController objects and emits
    /// gpibControllersAvailable() with their keys.
    void getActiveGpibControllers();

    // Hardware connection status queries

    /// \brief Returns whether all critical hardware objects are currently
    /// connected.
    /// \return \c true if every HardwareObject with d_critical set to \c true
    ///         reports isConnected().
    bool allCriticalHardwareConnected() const;

    // Dynamic hardware synchronization

    /// \brief Synchronizes d_hardwareMap with the current RuntimeHardwareConfig,
    /// adding, removing, or replacing hardware objects as needed.
    void syncWithRuntimeConfig();

    /// \brief Applies an explicit hardware map, reconciling additions, removals,
    /// and replacements against the current d_hardwareMap.
    /// \param map Target hardware map of \c "<Type>.<label>" → implementation keys.
    void applyHardwareMap(const std::map<QString, QString, std::less<>> &map);

    // Library configuration integration

    /// \brief Reapplies vendor library settings to all hardware objects whose
    /// driver depends on a runtime-configurable library.
    /// \return \c true if all affected objects accepted the updated library state.
    bool applyVendorLibraryChanges();

    /// \brief Triggers a Python script hot-reload for the hardware identified
    /// by \a hwKey and emits pythonScriptReloadResult() when complete.
    /// \param hwKey Hardware key of the Python-backed device to reload.
    void reloadPythonScript(const QString &hwKey);

public:
    /// \brief Returns the validation data keys reported by all hardware objects.
    /// \return Map of hardware key → list of validation data keys.
    std::map<QString,QStringList,std::less<>> validationKeys() const;

    /// \brief Resolves a GpibController by key and invokes \a callback with the
    /// result, holding the hardware map read-lock during the lookup.
    ///
    /// The callback receives a pointer to the controller, or \c nullptr if no
    /// controller with the given key exists. The callback is invoked synchronously
    /// on the calling thread while the read-lock is held; it must not attempt to
    /// acquire the write-lock.
    ///
    /// \param controllerKey Hardware key of the requested GpibController.
    /// \param callback Function called with the resolved controller pointer.
    void resolveGpibController(const QString& controllerKey, std::function<void(GpibController*)> callback) const;

    /// \brief Returns whether connection tests are currently in progress.
    /// \return \c true while at least one hardware object has not yet responded
    ///         to the most recent test round.
    bool connectionTestsInProgress() const { return d_connectionState.testsInProgress; }

private:
    /// \brief Tracks the progress of a connection-test round using lock-free
    /// atomic counters.
    ///
    /// reset() opens a new test round; recordResponse() counts each hardware
    /// object that reports back; allResponded() returns \c true when the count
    /// equals the expected number; markComplete() closes the round.
    /// Access to this struct is serialized by d_connectionStateLock.
    struct ConnectionTestState {
        std::atomic<size_t> responseCount{0}; ///< Responses received so far.
        std::atomic<bool> testsInProgress{false}; ///< Whether a round is open.

        /// \brief Begins a new test round, resetting the response counter.
        void reset() { responseCount = 0; testsInProgress = true; }
        /// \brief Records one hardware response.
        void recordResponse() { responseCount++; }
        /// \brief Returns \c true if \a expected responses have been received.
        bool allResponded(size_t expected) const { return responseCount >= expected; }
        /// \brief Marks the round as complete.
        void markComplete() { testsInProgress = false; }
    };
    ConnectionTestState d_connectionState;

    /// \brief Checks whether all hardware has responded and emits
    /// allHardwareConnected() if so.
    void checkStatus();

    /// \brief Initializes d_connectionState for a new test round.
    void initializeConnectionTesting();

    /// \brief Resets d_connectionState without marking a new round as open.
    void resetConnectionTestState();

    /// \brief Marks the current test round as complete.
    void finalizeConnectionTesting();

    // Constructor helper methods

    /// \brief Wires common signal connections from \a obj to the manager.
    void setupHardwareObject(HardwareObject* obj);

    // Runtime configuration integration

    /// \brief Creates a HardwareObject of the given type, implementation, and
    /// label via HardwareRegistry.
    /// \param type Hardware type class name.
    /// \param implementation Implementation key.
    /// \param label User-assigned label.
    /// \return Newly created object, or \c nullptr on failure.
    HardwareObject* createSpecificHardware(const QString& type, const QString& implementation, const QString& label);

    // Dynamic hardware synchronization internals

    /// \brief Removes the hardware object identified by \a hwKey from the map,
    /// disconnects its connections, and cleans up its thread.
    void removeHardwareInternal(const QString& hwKey);

    /// \brief Creates and installs the hardware object for \a hwKey using
    /// \a implementation.
    void addHardwareInternal(const QString& hwKey, const QString& implementation);

    /// \brief Replaces the hardware object for \a hwKey with one built from
    /// \a newImplementation.
    void replaceHardwareInternal(const QString& hwKey, const QString& newImplementation);

    // Synchronization orchestration helpers

    /// \brief Returns the set of hardware keys in d_hardwareMap that are absent
    /// from \a targetHardware and should be removed.
    std::vector<QString> findHardwareToRemove(const std::map<QString, QString, std::less<>>& targetHardware);

    /// \brief Returns hardware key / implementation pairs present in
    /// \a targetHardware but absent from d_hardwareMap and should be added.
    std::vector<std::pair<QString, QString>> findHardwareToAdd(const std::map<QString, QString, std::less<>>& targetHardware);

    /// \brief Returns hardware key / implementation pairs whose implementation
    /// has changed and should be replaced.
    std::vector<std::pair<QString, QString>> findHardwareToReplace(const std::map<QString, QString, std::less<>>& targetHardware);

    /// \brief Ensures every GPIB-dependent instrument has its controller
    /// resolved after a sync cycle.
    void resolveGpibControllersForInstruments();

    /// \brief Notifies ClockManager of any changes to the set of clock hardware
    /// objects after a sync cycle.
    void updateClockManager();

    // Library dependency tracking

    /// \brief Adds hardware objects whose vendor library configuration has
    /// changed to the removal / addition / replacement lists so they are
    /// recreated with the new library state.
    void addLibraryDependentHardwareToRecreation(const std::map<QString, QString, std::less<>>& targetHardware,
                                                std::vector<QString>& toRemove,
                                                std::vector<std::pair<QString, QString>>& toAdd,
                                                std::vector<std::pair<QString, QString>>& toReplace);

    // Connection tracking helpers

    /// \brief Stores a Qt connection handle keyed by \a hwKey so it can be
    /// cleanly disconnected when the object is removed.
    void storeConnection(const QString& hwKey, const QMetaObject::Connection& connection);

    /// \brief Wires common signal connections for \a obj and stores each
    /// connection handle in d_hardwareConnections.
    void setupHardwareObjectWithTracking(HardwareObject* obj);

    /// \brief Wires hardware-type-specific signal connections for \a obj and
    /// stores each handle in d_hardwareConnections.
    void setupHardwareSpecificConnectionsWithTracking(HardwareObject* obj);

    /// \brief Disconnects and removes all stored connections for \a hwKey.
    void disconnectStoredConnections(const QString& hwKey);

    /// \brief Live map from hardware key to owning HardwareObject pointer.
    ///
    /// Protected by d_hardwareMapLock. Keys use the \c "<Type>.<label>" format
    /// defined by BC::Key::parseKey().
    std::map<QString,HardwareObject*,std::less<>> d_hardwareMap;

    /// \brief Per-hardware-key lists of stored Qt connection handles used for
    /// clean disconnection when objects are removed dynamically.
    ///
    /// Accessed only on the HardwareManager thread (by storeConnection,
    /// setupHardwareObjectWithTracking, and disconnectStoredConnections);
    /// not protected by d_hardwareMapLock or any other lock.
    std::map<QString, QVector<QMetaObject::Connection>, std::less<>> d_hardwareConnections;

    /// \brief Clock subsystem manager; owned by HardwareManager, lives on the
    /// same thread.
    std::unique_ptr<ClockManager> pu_clockManager;

    /// \brief Raw pointer to the single live instance, set in the constructor
    /// and cleared in the destructor, used by constInstance().
    static HardwareManager* s_instance;

    /// \brief Read-write lock protecting d_hardwareMap.
    ///
    /// Readers (findHardware, findHardwareByType, checkStatus, validationKeys,
    /// resolveGpibController, getActiveGpibControllers,
    /// allCriticalHardwareConnected, testAll, testObjectConnection) acquire a
    /// read lock; writers (addHardwareInternal, removeHardwareInternal)
    /// acquire a write lock. storeAllOptHw acquires a read lock indirectly via
    /// the findHardware-based accessors it calls.
    mutable QReadWriteLock d_hardwareMapLock;

    /// \brief Mutex protecting d_connectionState.
    ///
    /// Held briefly while updating responseCount and testsInProgress; never
    /// held at the same time as d_hardwareMapLock to prevent deadlocks.
    mutable QMutex d_connectionStateLock;

    /// \brief Locates a hardware object by exact key and casts it to \c T*.
    ///
    /// Acquires a read lock on d_hardwareMapLock for the duration of the map
    /// lookup. Returns \c nullptr if the key is absent or the cast fails.
    ///
    /// \tparam T Target HardwareObject subtype.
    /// \param key Hardware key to look up.
    /// \return Pointer to the cast object, or \c nullptr.
    template<class T>
    T* findHardware(const QString key) const {
        QReadLocker locker(&d_hardwareMapLock);
        auto it = d_hardwareMap.find(key);
        if (it == d_hardwareMap.end()) {
            return nullptr;
        }

        return qobject_cast<T*>(it->second);
    }

    /// \brief Returns all hardware objects of type \c T currently in the map.
    ///
    /// Acquires a read lock on d_hardwareMapLock and iterates the entire map,
    /// comparing each entry's type prefix against \c T::staticMetaObject.className().
    /// Uses qobject_cast for safety; entries that fail the cast are skipped.
    ///
    /// \tparam T Target HardwareObject subtype.
    /// \return Vector of all matching objects in map-iteration order.
    template<class T>
    QVector<T*> findHardwareByType() const {
        QReadLocker locker(&d_hardwareMapLock);

        QVector<T*> result;

        for (const auto& [key,hwObj] : d_hardwareMap) {
            QString hwType = BC::Key::parseKey(key).first;
            if (hwType == T::staticMetaObject.className()) {
                if (T* hw = qobject_cast<T*>(hwObj)) {
                    result.append(hw);
                }
            }
        }

        return result;
    }

};

#endif // HARDWAREMANAGER_H
