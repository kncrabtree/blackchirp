#ifndef HARDWAREOBJECT_H
#define HARDWAREOBJECT_H

#include <QObject>
#include <QString>
#include <QSettings>
#include <QCoreApplication>
#include <QList>
#include <QPair>

#include <data/loghandler.h>
#include <data/storage/settingsstorage.h>
#include <data/storage/auxdatastorage.h>
#include <hardware/core/communication/communicationprotocol.h>
#include <hardware/core/communication/virtualinstrument.h>
#include <hardware/core/communication/tcpinstrument.h>
#include <hardware/core/communication/rs232instrument.h>
#include <hardware/core/communication/custominstrument.h>
#include <hardware/core/communication/gpibinstrument.h>

#include <data/experiment/experiment.h>


/*!
 * \brief Abstract base class for all hardware connected to the instrument.
 *
 * Every device Blackchirp talks to inherits from HardwareObject. The class
 * provides the common identity, communication, settings, and lifecycle
 * machinery that the rest of the program relies on; concrete drivers add
 * the per-vendor command translation and any device-specific behavior.
 *
 * # Identity
 *
 * Each instance has a *hardware type* (e.g., \c "AWG",
 * \c "FtmwScope", \c "PulseGenerator") naming the abstract role and
 * a *label* — a free-form, user-supplied string distinguishing
 * multiple instances of the same type (e.g., \c "main",
 * \c "secondary"). The hardware type matches the interface class's
 * metaobject name by convention.
 *
 * Type and label combine to form d_key (e.g.,
 * \c "PulseGenerator.main"), which uniquely identifies the instance.
 * d_key is also the SettingsStorage group that holds this instance's
 * persistent settings, which is why the same key cannot be reused
 * for two different drivers: an instance's driver is fixed at
 * profile creation, and changing it requires deleting the profile
 * and rebuilding it under the same (or a new) label.
 *
 * d_model is a separate string carrying just the driver class name
 * (e.g., \c "AWG70002a", \c "PythonAwg"); it is recorded inside the
 * settings group and used for display purposes — for example, to
 * show the user which driver backs a profile in the hardware
 * settings dialog. d_model is not part of d_key and is not used as
 * a dispatch identifier.
 *
 * # Constructing a driver
 *
 * Two-layer inheritance is the convention: an *interface* class
 * (\c AWG, \c FtmwDigitizer, \c PulseGenerator, ...) declares the slot
 * and signal API the rest of Blackchirp consumes, and one or more
 * *driver* classes (\c AWG70002a, \c VirtualAwg, ...) translate that
 * API into vendor commands. A typical pair:
 *
 *     class Analyzer : public HardwareObject
 *     {
 *         Q_OBJECT
 *     public:
 *         Analyzer(const QString& impl, const QString& label, QObject *parent = nullptr)
 *             : HardwareObject(QString(Analyzer::staticMetaObject.className()),
 *                              impl, label, parent) {}
 *
 *     public slots:
 *         bool setFoo(double f) {
 *             auto ok = hwSetFoo(f);
 *             if(!ok) emit hardwareFailure();
 *             else    readFoo();
 *             return ok;
 *         }
 *         double readFoo() {
 *             d_foo = hwReadFoo();
 *             emit fooUpdated(d_foo, QPrivateSignal());
 *             return d_foo;
 *         }
 *
 *     signals:
 *         void fooUpdated(double, QPrivateSignal);
 *
 *     private:
 *         double d_foo;
 *         virtual bool   hwSetFoo(double f) = 0;
 *         virtual double hwReadFoo()        = 0;
 *     };
 *
 *     class VendorAnalyzer : public Analyzer
 *     {
 *     public:
 *         VendorAnalyzer(const QString& label, QObject *parent = nullptr)
 *             : Analyzer(QString(VendorAnalyzer::staticMetaObject.className()),
 *                        label, parent) {}
 *     private:
 *         bool   hwSetFoo(double f) override;
 *         double hwReadFoo()        override;
 *     };
 *
 * Each driver is also registered at static initialization with
 * \c REGISTER_HARDWARE_META, \c REGISTER_HARDWARE_PROTOCOLS, and
 * (where the driver introduces settings) \c REGISTER_HARDWARE_SETTINGS;
 * see hardwareregistration.h.
 *
 * # Configuration flags
 *
 * Drivers may set the following flags in their constructors:
 *
 * - \c d_threaded — if true, the HardwareManager moves the object to a
 *   dedicated QThread. Threaded objects must not have a QObject parent
 *   and must not create child QObjects in their constructors;
 *   construct children inside initialize() instead.
 * - \c d_critical — if true (the default), a hardwareFailure() emission
 *   aborts any active experiment and blocks new ones until
 *   bcTestConnection() succeeds again. If false, the object is simply
 *   marked disconnected and the experiment continues. The flag is also
 *   user-editable through the hardware settings dialog.
 * - \c d_commType — the active CommunicationProtocol type. The set of
 *   protocols a driver supports is declared via
 *   \c REGISTER_HARDWARE_PROTOCOLS; the user picks one at profile
 *   creation time and the persisted value is loaded by the base
 *   constructor.
 *
 * Drivers whose CommunicationProtocol type is
 * CommunicationProtocol::Custom additionally declare any user-supplied
 * connection parameters (device path, serial number, etc.) following
 * the convention documented in CustomInstrument.
 *
 * # Lifecycle
 *
 * HardwareObject instances are not bound to program lifetime. The
 * HardwareManager creates and destroys them in response to loadout
 * activation, profile edits, and protocol changes. After construction
 * the manager calls bcInitInstrument(), which builds the
 * CommunicationProtocol object, calls initialize(), and wires up
 * hardwareFailure routing. Drivers must implement initialize(): it is
 * the place to construct child QObjects (the constructor cannot, for
 * threaded drivers) and perform any one-shot setup. Per-connection
 * work belongs in testConnection() instead, since connection attempts
 * happen many times over the lifetime of an instance.
 *
 * # Connection testing
 *
 * Drivers must implement testConnection(): it should attempt some
 * cheap interaction with the device (typically an \c *IDN? query) to
 * confirm both that something is responding and that it is the
 * expected hardware. On failure, store a descriptive message in
 * \c d_errorString and return false. testConnection() is called from
 * the bcTestConnection() wrapper, which first reloads settings from
 * SettingsStorage; the override therefore sees up-to-date settings
 * and may read them freely.
 *
 * # Optional virtual hooks
 *
 * Drivers may override these hooks to participate in the experiment
 * lifecycle and the auxiliary-data system:
 *
 * - validationKeys() — keys whose values appear on the experiment
 *   setup dialog's Validation page.
 * - sleep() — switch the device to/from a low-power state.
 * - beginAcquisition() / endAcquisition() — start- and end-of-experiment
 *   hooks.
 * - prepareForExperiment() — validate and stage per-experiment
 *   settings; called via the hwPrepareForExperiment() wrapper.
 * - readAuxData() — values to plot on the Aux/Rolling tabs and to
 *   persist as auxiliary data.
 * - readValidationData() — values to verify post-acquisition, separate
 *   from the plotted aux data.
 * - readSettings() — refresh cached state when the user accepts the
 *   hardware settings dialog.
 *
 * # Threading
 *
 * Functions intended to be invoked from outside the object — for
 * example by HardwareManager during an acquisition — must be declared
 * as Qt slots so they reach the object via queued connection when the
 * driver is threaded.
 */
class HardwareObject : public QObject, public SettingsStorage
{
	Q_OBJECT
	friend class HardwareCommunicationTest;
public:

    /*!
     * \brief Constructor.
     *
     * Combines \a hwType and \a label into d_key (the unique
     * identifier and SettingsStorage group for this instance), stores
     * \a hwImpl in d_model for display, loads the persisted
     * critical/protocol values, and applies registry-supplied setting
     * defaults.
     *
     * \param hwType Hardware type, by convention the interface class's
     * metaobject name (e.g., \c "FtmwScope").
     * \param hwImpl Driver class name, stored in d_model for display
     * (e.g., \c "VirtualFtmwScope").
     * \param label User-supplied label distinguishing instances of the
     * same type.
     * \param parent QObject parent. Must be \c nullptr if the driver
     * sets d_threaded to true in its constructor.
     */
    explicit HardwareObject(const QString& hwType, const QString& hwImpl, const QString& label, QObject *parent = nullptr);
    virtual ~HardwareObject();

    const QString d_key;   /*!< Unique identifier and SettingsStorage group: "hwType.label". Fixed at profile creation. */
    const QString d_model; /*!< Driver class name, used for display only (e.g., "AWG70002a"). */

    bool d_critical; /*!< Whether a communication error should abort an experiment. */
    bool d_threaded; /*!< Whether the object should have its own thread of execution. */
    CommunicationProtocol::CommType d_commType; /*!< Type of communication */

    /*!
     * \brief Returns most recent error message
     * \return Error message
     */
    QString errorString();

    /*!
     * \brief Returns whether the instrument is currently connected.
     *
     * Does not attempt to communicate with the device; returns the cached
     * d_isConnected value, which is set to true when testConnection()
     * returns true and reset to false when hardwareFailure() is emitted.
     *
     * \sa bcTestConnection
     *
     * \return Whether the device is connected.
     */
    bool isConnected() const { return d_isConnected; }

    /*!
     * \brief Returns list of validation keys
     * 
     * Validation keys indicate settings which can be checked on the Validation
     * page of the experiment setup dialog. This function should be overriden
     * to return keys that will be used with the readValidationData() function.
     * 
     * The default implementation returns an empty list.
     * 
     * \sa readValidationData
     * 
     * \return List of validation key strings.
     */
    virtual QStringList validationKeys() const { return {}; }

    /*!
     * \brief Removes all persisted settings for this hardware profile
     *
     * Calls SettingsStorage::purge() to delete the entire QSettings group
     * for this hardware object. Should be called before deleteLater() when
     * permanently removing a hardware profile.
     */
    void purgeSettings() { purge(); }

    /*!
     * \brief Returns the list of supported communication protocols.
     *
     * Looks the protocol set up in HardwareRegistry by this instance's
     * hardware type and implementation key. The set is populated at
     * static-initialization time by \c REGISTER_HARDWARE_PROTOCOLS;
     * change a driver's supported protocols by editing that macro
     * invocation in the driver's .cpp file, not by overriding this
     * function. Returns \c {CommunicationProtocol::Virtual} when no
     * protocols are registered for the implementation.
     *
     * \return Vector of supported CommunicationProtocol::CommType values.
     */
    QVector<CommunicationProtocol::CommType> supportedProtocols() const;
	
signals:
    /*!
     * \brief Indicates whether a connection is successful
     * \param success True if connection is successful
     * \param msg If unsuccessful, an optional message to display on the log tab along with the failure message.
     */
    void connected(bool success,QString msg,QPrivateSignal);

    /*!
     * \brief Signal emitted when communication with the device fails.
     *
     * Emitting this signal sets d_isConnected to false. If d_critical
     * is true the active experiment is aborted and new experiments are
     * blocked until bcTestConnection() succeeds.
     */
    void hardwareFailure();

    /*!
     * \brief Signal containing data for Aux Data plots.
     *
     * The AuxDataStorage::AuxDataMap payload holds the key/value pairs for
     * Aux Data.
     *
     * \sa readAuxData
     */
    void auxDataRead(AuxDataStorage::AuxDataMap,QPrivateSignal);

    /*!
     * \brief Signal containing data for experiment validation.
     *
     * The AuxDataStorage::AuxDataMap payload holds the key/value pairs for
     * experiment validation.
     *
     * \sa readAuxData, validationKeys
     */
    void validationDataRead(AuxDataStorage::AuxDataMap,QPrivateSignal);

    /*!
     * \brief Signal containing data for Rolling Data plots.
     *
     * This signal is emitted on a timer whose timeout interval is controlled
     * by the user. When the timer fires, readAuxData() is called and the
     * results are emitted in this signal as an AuxDataStorage::AuxDataMap of
     * key/value pairs.
     *
     * \sa timerEvent, readAuxData
     */
    void rollingDataRead(AuxDataStorage::AuxDataMap,QPrivateSignal);
	
public slots:
    /*!
     * \brief Wrapper for one-shot initialization after construction.
     *
     * Called by the HardwareManager after the object has been moved to
     * its thread (if d_threaded is true) and the thread has started.
     * It:
     *
     * -# builds the CommunicationProtocol via buildCommunication() and
     *    calls its initialize();
     * -# calls the driver's initialize() override;
     * -# wires the hardwareFailure() signal to clear d_isConnected.
     *
     * Connection testing is dispatched separately via bcTestConnection().
     *
     * \sa initialize
     */
    void bcInitInstrument();
    
    /*!
     * \brief Wrapper for testing hardware communication.
     *
     * Called from the HardwareManager when the user requests a
     * connection test from the UI, and from the connection-testing
     * pass that follows hardware creation. Reloads settings from disk
     * via bcReadSettings(), tests the CommunicationProtocol's
     * underlying QIODevice, then calls the driver's testConnection()
     * override. The result is stored in d_isConnected and reported via
     * the connected() signal.
     *
     * \sa testConnection
     */
    void bcTestConnection();
    
    /*!
     * \brief Wrapper for reading aux and validation data.
     *
     * Calls readAuxData() and readValidationData(), emitting auxDataRead()
     * and validationDataRead() with whatever each returns. No-op if the
     * device is not connected.
     *
     * \sa readAuxData, readValidationData
     */
    void bcReadAuxData();

    /*!
     * \brief Wrapper for reloading settings from disk.
     *
     * Called whenever the user closes the hardware settings dialog or
     * a connection test is initiated. Refreshes the in-memory
     * SettingsStorage cache, updates d_critical and d_commType,
     * restarts the rolling-data timer, and then calls readSettings()
     * so the driver can refresh its own cached state.
     *
     * \sa readSettings
     */
    void bcReadSettings();

    /*!
     * \brief Puts device into a standby mode.
     * 
     * The default implementation does nothing.
     * 
     * \param b If true, go into standby mode. Else, active mode.
     */
	virtual void sleep(bool b);
    
    /*!
     * \brief Per-experiment preparation wrapper.
     *
     * If the device is currently disconnected, reattempts the
     * connection; if that still fails and d_critical is true, marks
     * the experiment with an error and returns false. Otherwise
     * delegates to prepareForExperiment(). Interface classes override
     * this when they need to perform setup that runs even for
     * non-critical disconnected devices (registering keys, writing
     * derived settings, etc.).
     *
     * \todo Split into two virtuals so interface classes do not have
     * to override both this and prepareForExperiment().
     *
     * \sa prepareForExperiment
     *
     * \param exp Experiment carrying the requested per-device settings;
     * may be mutated to record actual settings or an error string.
     * \return Whether preparation succeeded.
     */
    virtual bool hwPrepareForExperiment(Experiment &exp);
    
    /*!
     * \brief Hook called when an experiment begins.
     *
     * Drivers may override to start acquisition (for example, an AWG
     * starts playing waveforms; an FtmwDigitizer arms for trigger
     * events). The default implementation does nothing.
     */
    virtual void beginAcquisition(){}
    
    /*!
     * \brief Hook called when an experiment ends.
     *
     * Called whether the experiment completed normally or was aborted.
     * Drivers may override to stop acquisition (for example, halting
     * AWG playback). The default implementation does nothing.
     */
    virtual void endAcquisition(){}

    /*!
     * \brief Sets communication protocol at runtime
     * 
     * Allows changing the communication protocol after construction.
     * The protocol must be supported as declared by supportedProtocols().
     * This will rebuild the communication object with the new protocol.
     * 
     * \param commType New communication protocol type
     * \param gc Pointer to the `GpibController` object for GPIB devices
     * \return True if protocol change was successful
     */
    bool setCommProtocol(CommunicationProtocol::CommType commType, QObject *gc = nullptr);

    /*!
     * \brief (Re)build the CommunicationProtocol object.
     *
     * Constructs the CommunicationProtocol subclass that matches
     * \a commType (or d_commType if \a commType is
     * CommunicationProtocol::None) into p_comm, replacing any prior
     * instance, and forwards p_comm's hardwareFailure() to this
     * object's hardwareFailure().
     *
     * \param gc Pointer to the GpibController, used only when the
     * selected protocol is GPIB.
     * \param commType New protocol type, or
     * CommunicationProtocol::None to use the persisted d_commType.
     */
    void buildCommunication(QObject *gc = nullptr, CommunicationProtocol::CommType commType = CommunicationProtocol::None);

protected:
    QString d_errorString;         /*!< Last error message; consumed by errorString(). */
    bool d_enabledForExperiment;   /*!< Whether the device is active in the current experiment. */
    CommunicationProtocol *p_comm; /*!< Active communication protocol; built by buildCommunication(). */

    /// Log a message to the global LogHandler with this device's d_key prepended.
    void hwLog(QAnyStringView text)   { bcLog(u"%1: %2"_s.arg(d_key, text));   }
    /// Log a warning to the global LogHandler with this device's d_key prepended.
    void hwWarn(QAnyStringView text)  { bcWarn(u"%1: %2"_s.arg(d_key, text));  }
    /// Log an error to the global LogHandler with this device's d_key prepended.
    void hwError(QAnyStringView text) { bcError(u"%1: %2"_s.arg(d_key, text)); }
    /// Log a debug message to the global LogHandler with this device's d_key prepended.
    void hwDebug(QAnyStringView text) { bcDebug(u"%1: %2"_s.arg(d_key, text)); }

    /*!
     * \brief Validate and stage per-experiment settings.
     *
     * Drivers override this to read the requested per-device settings
     * out of \a exp, push them to the device, and write back the
     * settings actually applied (which may differ from what was
     * requested). On failure, store a descriptive message in
     * \c exp.d_errorString and return false. The default implementation
     * returns true.
     *
     * \sa hwPrepareForExperiment
     *
     * \param exp Mutable Experiment carrying per-device settings.
     * \return Whether preparation succeeded.
     */
    virtual bool prepareForExperiment(Experiment &exp) { Q_UNUSED(exp) return true; }

    /*!
     * \brief One-shot initialization run after construction.
     *
     * Drivers must implement this even if there is nothing to do.
     * It is the place to construct child QObjects (the constructor
     * cannot, for threaded drivers) and perform any setup that should
     * run once per instance lifetime. Per-connection work belongs in
     * testConnection() instead.
     *
     * \sa bcInitInstrument
     */
    virtual void initialize() =0;

    /*!
     * \brief Attempt to communicate with the device.
     *
     * Drivers must implement this. Send a cheap test interaction
     * (typically an \c *IDN? query) to confirm the device is responsive
     * and is the expected hardware. On failure, store a descriptive
     * message in d_errorString.
     *
     * \sa bcTestConnection
     *
     * \return Whether the connection attempt succeeded.
     */
    virtual bool testConnection() =0;


private:
    /*!
     * \brief Apply registered setting defaults from HardwareRegistry
     *
     * Called from the base constructor to apply any settings registered
     * via REGISTER_HARDWARE_SETTINGS and REGISTER_HARDWARE_ARRAY macros.
     * Uses setDefault() for scalar settings (preserves existing values)
     * and setArray() for array settings (only if array doesn't exist yet).
     *
     * \param hwType Hardware type key (e.g., "FtmwScope", "AWG")
     */
    void applyRegisteredSettings(const QString& hwType);

    /*!
     * \brief Perform read of auxiliary data
     *
     * Derived classes may override this function to return data to be
     * displayed on the aux and/or rolling data plots.
     *
     * \return AuxDataStorage::AuxDataMap (an alias for `std::map<QString,QVariant,std::less<>>`)
     * containing key-value pairs.
     */
    virtual AuxDataStorage::AuxDataMap readAuxData();
    
    /*!
     * \brief Perform read of validation data
     * 
     * Derived classes may override this function to return any additional data
     * that can be used for experiment validation which is **not** already
     * returned in readAuxData(). This may, for example, consist of digital
     * data that is not appropriate for graphical presentation.
     * 
     * \return AuxDataStorage::AuxDataMap (an alias for `std::map<QString,QVariant,std::less<>>`)
     * containing key-value pairs.
     */
    virtual AuxDataStorage::AuxDataMap readValidationData();
    
    /*!
     * \brief Update values from SettingsStorage
     * 
     * Derived classes may override this function to read and process any
     * updates to settings made by the user in the Hardware Settings dialog.
     * The function is called automatically when the user accepts that dialog
     * and also when bcTestConnection() is called.
     * 
     */
    virtual void readSettings() {}

    bool d_isConnected; /*!< Contains whether device's last communication was successful. */
    int d_rollingDataTimerId{-1}; /*!< ID for rolling data timerEvent */

    // QObject interface
protected:
    
    /*!
     * \brief Rolling-data timer handler.
     *
     * On each tick of the rolling-data timer (configured by the user
     * via BC::Key::HW::rInterval), calls readAuxData() and emits
     * rollingDataRead() with the result.
     *
     * \param event Timer event that triggered the call.
     */
    void timerEvent(QTimerEvent *event) override;
};

#endif // HARDWAREOBJECT_H
