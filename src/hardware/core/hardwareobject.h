#ifndef HARDWAREOBJECT_H
#define HARDWAREOBJECT_H

#include <QObject>
#include <QString>
#include <QSettings>
#include <QApplication>
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
 * Keys used to refer to settings in SettingsStorage
 */
namespace BC::Key::HW {
static const QString key{"key"}; /*!< Hardware type key */
static const QString subKey{"subKey"}; /*!< Hardware implementation key */
static const QString name{"prettyName"}; /*!< Display name */
static const QString connected{"connected"}; /*!< Whether last communication was successful */
static const QString critical{"critical"}; /*!< Whether communication failure should abort an experiment */
static const QString threaded{"threaded"}; /*!< Whether object is in its own thread */
static const QString commType{"commType"}; /*!< CommunicationProtocol type */
static const QString rInterval{"rollingDataIntervalSec"}; /*!< Timer interval for rolling data (seconds) */
}

/*!
 * \brief Abstract base class for all hardware connected to the instrument.
 *
 * This class establishes a common interface for all hardware. Adding a new
 * piece of hardware to the code involves creating a subclass of
 * HardwareObject. Each hardware object has a name (::d_name) that is used in
 * the user interface, and a key (::d_key) and subkey (::d_subKey) that are
 * used to refer to the object in the settings file. The ::d_key specifies a
 * particular type of HardwareObject, and the ::d_subKey specifies a particular
 * implementation of that hardware type. Subclasses must assign strings to
 * these variables in their constructors and must also set the
 * CommunicationProtocol in the class constructor. Other options, including
 * whether the HardwareObject is pushed to its own thread of execution or
 * whether the hardware is critical to program operation may be optionally set
 * as well. Finally, each HardwareObject has a ::d_index which is fused with
 * its ::d_key to distinguish objects of the same type. The ::d_index value is
 * initialized in the constructor and should be assigned using a static
 * variable in the derived class for the hardware type which is incremented in
 * its constructor. This ensures that each object of that type is given a
 * unique index corresponding to its order of construction. Example:
 * 
 *     class NewHwType : public HardwareObject
 *     {
 *     public:
 *         NewHwType(QString subKey, QObject *parent) :
 *           HardwareObject("newType",subKey,"Name",CommunicationProtocol::Rs232,
 *                          parent,false,false,d_newIndex) { d_newIndex++; }
 *                           
 *     private:
 *         inline static int d_newIndex = 0;
 *     }
 * 
 * For instruments that do not communicate by GPIB, RS232, or TCP (i.e., a
 * CustomInstrument), a general interface is available to allow the user to
 * specify any information needed to connect to the device (file handles, ID
 * numbers, etc). In the constructor of the implementation, create a
 * SettingsStorage array with the key BC::Key::Custom::comm. Each entry must
 * define the following parameters:
 * 
 * - BC::Key::Custom::label: A QString that will be displayed to the user to
 * indicate what information is sought (e.g., Device name, ID number, etc)
 * - BC::Key::Custom::key: A QString that the implementation will use to
 * retrieve the entered value type.
 * - BC::Key::Custom::type: Type of data entry to expose on UI. Possible values:
 *     - BC::Key::Custom::stringKey: Creates a text box for string.
 *     - BC::Key::Custom::intKey: Creates a numeric integer spinbox.
 * 
 * In addition, the following optional settings may be made:
 * 
 * - BC::Key::Custom::maxLen: Maximum length of string input.
 * - BC::Key::Custom::intMin: Lower limit on numeric entries (defaut: -2^31)
 * - BC::Key::Custom::intMax: Upper limit on numeric entries (default: 2^31-1)
 *
 *
 * A HardwareObject will be moved to its own thread if the ::d_threaded varialbe
 * is set to true in the constructor. If so, the object must not be assigned a
 * parent, and any QObjects that will be created in the child class should NOT
 * be initialized in the constructor (see initialize()).
 * 
 * The ::d_critical variable (also set in the constructor) determines whether the
 * hardware is deemed essential for instrument operation. When a critical piece
 * of hardware experiences a failure and emits the hardwareFailure() signal,
 * any ongoing acquisition is terminated and no new acquisitions can be
 * initiated until communication is re-established. If a non-critical object
 * has a failure, it is disconnected and subsequent operations are ignored
 * until communication is re-established, and any ongoing acquisitions are
 * allowed to proceed.
 * 
 * Implementations of HardwareObject must implement two functions: initialize()
 * and testConnection(). The initialize() function is called after a
 * HardwareObject is created (and pushed into its own thread, if appropriate).
 * Any necessary QObject children should be created within the initialize()
 * function, and any other one-time initialization tasks may be performed here
 * as well. The only time initialize() is called is upon program startup, so
 * any tasks that need to be done per-connection should be implemented in
 * testConnection().
 * 
 * In the testConnection() implementation, the derived class should attempt to
 * establish communication with the physical device and perform some sort of
 * test to make sure that the connection works and that the correct device is
 * targeted. Usually, this takes the form of an ID query of some type (e.g.,
 * `*IDN?`). If unsuccessful, a message should be stored in d_errorString, but
 * messages may also be emitted to the UI using the logMessage() signal. The
 * testConnection() function must return a bool indicating whether the
 * connection was successful. Note: testConnection() is called from the
 * bcTestConnection() wrapper function which reloads settings from
 * SettingsStorage prior to making the testConnection() call. Inside
 * testConnection() therefore, all settings have already been updated and may
 * be read from.
 * 
 * The HardwareObject base class provides several additional virtual functions
 * that may be reimplemented in derived classes to perform various operations:
 * 
 * - validationKeys()
 * - sleep()
 * - forbiddenKeys()
 * - beginAcquisition()
 * - endAcquisition()
 * - prepareForExperiment()
 * - readAuxData()
 * - readValidationData()
 * - readSettings()
 * 
 * Any functions that need to be called from outside the class (e.g., from the
 * HardwareManager during a scan) should be declared as slots so that they
 * can be invoked using Qt's queued connection mechanism.
 * 
 * When creating a new HardwareObject type to add to the program, the derived
 * class should define a new ::d_key and establish an interface for more
 * specific implementations (i.e., different manufactuers/models). The main
 * functionality which interoperates with the rest of the program should be
 * defined using signals and slots so that the device may work in its own
 * thread of execution if desired. For each setting or group of settings (as
 * appropriate), the interface class should define a wrapper function that can
 * be called from other Blackchirp code and a pure virtual function that the
 * implementation classes define to perform the actual communication. For
 * example, if a new hardware type Analyzer is to be added, and its property
 * `d_foo` needs to be settable and readable, a skeleton implementation may
 * look like this:
 * 
 *     class Analyzer : public HardwareObject
 *     {
 *         //other functions, etc...
 *         
 *     public slots:
 *         bool setFoo(double f) {
 *             auto out = hwSetFoo(f);
 *             if(!out)
 *                 emit hardwareFailure();
 *             else
 *                 readFoo();
 *             return out;
 *         }
 *         
 *         double readFoo() {
 *             d_foo = hwReadFoo();
 *             emit fooUpdated(d_foo,QPrivateSignal());
 *             return d_foo;
 *         }
 *         
 *     signals:
 *         void fooUpdated(double,QPrivateSignal);
 *         
 *     private:
 *         double d_foo;
 *         
 *         // These functions would need to be implemented in a class that
 *         // inherits Analyzer
 *         virtual bool hwSetFoo() =0;
 *         virtual double hwReadFoo() =0;
 *     }
 * 
 * It is recommended to look at the patterns used in current HardwareObject
 * types to see how to create new types.
 */
class HardwareObject : public QObject, public SettingsStorage
{
	Q_OBJECT
public:
    /*!
     * \brief Constructor. Does nothing.
     *
     * \param parent Pointer to parent QObject. Should be 0 if it will be in its own thread.
     */
    explicit HardwareObject(const QString hwType, const QString subKey, const QString name,
                            CommunicationProtocol::CommType commType, QObject *parent = nullptr,
                            bool threaded = true, bool critical = true, int index=0);
    virtual ~HardwareObject();

    QString d_name; /*!< Name to be displayed on UI */
    const QString d_key; /*!< Name to be used in settings for abstract hardware. */
    const QString d_subKey; /*!< Name to be used in settings for real hardware. */
    const int d_index; /*!< Index used if multiple objects of same type are present. */

    bool d_critical; /*!< Whether a communication error should abort an experiment. */
    const bool d_threaded; /*!< Whether the object should have its own thread of execution. */
    const CommunicationProtocol::CommType d_commType; /*!< Type of communication */

    /*!
     * \brief Returns most recent error message
     * \return Error message
     */
    QString errorString();

    /*!
     * \brief Returns if the instrument is connected.
     * 
     * This function does not attempt to communicate with the device; it simply
     * returns the value of ::d_isConnected. That variable is set to true when
     * testConnection() returns true, and is set to false when
     * hardwareFailure() is emitted.
     * 
     * \sa bcTestConnection
     * 
     * \return 
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
     * \return List of valdidation key strings.
     */
    virtual QStringList validationKeys() const { return {}; }
	
signals:
    /*!
     * \brief Displays a message on the log.
     * \param QString The message to display
     * \param Blackchirp::MessageCode The status incidator (Normal, Warning,
     * Error, Highlight, Debug)
     */
	void logMessage(const QString, const LogHandler::MessageCode = LogHandler::Normal);

    /*!
     * \brief Indicates whether a connection is successful
     * \param bool True if connection is successful
     * \param msg If unsuccessful, an optional message to display on the log tab along with the failure message.
     */
    void connected(bool success,QString msg,QPrivateSignal);

    /*!
     * \brief Signal emitted if communication to hardware fails.
     * 
     * When this signal is emitted, ::d_isConnected is set to false. If
     * ::d_critical is true, then any ongoing experiment is aborted and new
     * experiments cannot be started until a successful call to
     * testConnection() is made.
     * 
     */
    void hardwareFailure();

    /*!
     * \brief Signal containing data for Aux Data plots
     * \param AuxDataStorage::AuxDataMap Keys and values for Aux Data
     * 
     * \sa readAuxData
     */
    void auxDataRead(AuxDataStorage::AuxDataMap,QPrivateSignal);
    
    /*!
     * \brief Signal containing data for experiment validation
     * \param AuxDataStorage::AuxDataMap Keys and values for experiment validation
     * 
     * \sa readAuxData, validationKeys
     */
    void validationDataRead(AuxDataStorage::AuxDataMap,QPrivateSignal);
    
    /*!
     * \brief Signal containing data for Rolling Data plots
     * 
     * This signal is emitted on a timer whose timeout interval is controlled
     * by the user. When the timer fires, readAuxData() is called and the
     * results are emitted in this signal.
     * 
     * \sa timerEvent, readAuxData
     * 
     * \param AuxDataStorage::AuxDataMap Keys and values for Rolling Data
     */
    void rollingDataRead(AuxDataStorage::AuxDataMap,QPrivateSignal);
	
public slots:
    /*!
     * \brief Wrapper for initializtion at program startup
     * 
     * This function is called after the object is moved to its thread and the
     * thread is started (if appropriate). It does the following:
     * 
     * -# Initializes the `CommunicationProtocol`.
     * -# Calls the initialize() virtual function.
     * -# Calls bcTestConnection().
     * -# Causes the hardwareFailure() signal to set ::d_isConnected to false.
     * 
     * \sa initialize
     */
    void bcInitInstrument();
    
    /*!
     * \brief Wrapper for testing hardware communication
     * 
     * This function is called first from bcInitInstrument() and then
     * subsequently when the user requests a connection test from the UI. First
     * the settings are read from disk, then the `CommunicationProtocol` and
     * instrument connections are tested. The result of testConnection() is
     * stored, and the connected() signal is emitted with the status message.
     * 
     * \sa testConnection
     */
    void bcTestConnection();
    
    /*!
     * \brief Wrapper function for reading aux data.
     * 
     * Calls both readAuxData() and readValidationData(), and emits the
     * auxDataRead() and validationDataRead() signals with the received data.
     * 
     * \sa readAuxData, readValidationData
     * 
     */
    void bcReadAuxData();

    /*!
     * \brief Wrapper function for reading from SettingsStorage
     * 
     * This function forces a reload of all settings from disk. It is called
     * whenever the user closes the Hardware Settings dialog box corresponding
     * to this object. It updates ::d_name and ::d_critical, and it restarts
     * the rolling data timer. It then calls readSettings() so that derived
     * classes can refresh any necessary settings.
     * 
     * \sa readSettings
     * 
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
     * \brief Wrapper function for initialization.
     * 
     * Attempts to reconnect if an instrument is disconnected, then calls
     * prepareForExperiment(). May be overridden in an interface class if
     * actions need to be taken upon successful initialization (registering
     * keys, writing settings, etc).
     * 
     * \todo Should create another virtual function for this; the logic of
     * overriding both this function and prepareForExperiment() is not good.
     * 
     * \sa prepareForExperiment
     * 
     * \param exp `Experiment` object containing hardware settings, etc
     * \return Whether initialization was successful
     */
    virtual bool hwPrepareForExperiment(Experiment &exp);
    
    /*!
     * \brief Returns list of settings keys which should not be user-adjustable.
     * 
     * If any settings should not be changed by the user, this function can be
     * overriden to return a list of the keys corresponding to those settings.
     * The default implementation returns an empty list. Note that settings
     * defined by HardwareObject are excluded automatically with the exceptions
     * of #BC::Key::HW::critical and BC::Key::HW::rInterval.
     * 
     * \return List of keys to exclude
     */
    virtual QStringList forbiddenKeys() const { return {}; }
    
    /*!
     * \brief Function called when experiment begins.
     * 
     * Derived classes may override this function to perform any actions to
     * initiate the data acquisition process after the experiment has been
     * fully initialized. For example, an `AWG` may begin playing waveforms,
     * and an `FtmwDigitizer` may begin querying for trigger events.
     * 
     * The default implementation does nothing.
     */
    virtual void beginAcquisition(){}
    
    /*!
     * \brief Function called when experiment ends.
     * 
     * Derived classes may override this function to perform any actions at the
     * end of an experiment, whether it is aborted or completed normalls. For
     * example, an `AWG` may stop its waveform playback.
     * 
     */
    virtual void endAcquisition(){}

    /*!
     * \brief Creates `CommunicationProtocol` object
     * 
     * Uses the `d_comm` parameter passed in the constructor to create the
     * appropriate `CommunicationProtocol` subclass (::p_comm), and ties together the
     * logMessage() and hardwareFailure() signals from each.
     * 
     * \param gc Pointer to the `GpibController` object for GPIB devices.
     * If the device is not GPIB, this variable is unused.
     */
    void buildCommunication(QObject *gc = nullptr);

protected:
    QString d_errorString; /*!< Last error. */
    bool d_enabledForExperiment; /*!< Whether the device is active in the current experiment. */
    CommunicationProtocol *p_comm; /*!< `QIODevice` subclass used for communication */

    /*!
     * \brief Initializes hardware before experiment.
     * 
     * This function should validate the settings stored in the experiment for
     * the hardware object and ensure the hardware is configured appropriately
     * as the user has requested. The `exp` object is mutable, and should be
     * updated with any actual settings of the hardware which may differ from
     * what the user requested. If initialization is unsuccessful, then store
     * an error message in `exp.d_errorString` and return false.
     * 
     * The default implementation returns true.
     * 
     * \sa hwPrepareForExperiment
     * 
     * \param exp
     * \return 
     */
    virtual bool prepareForExperiment(Experiment &exp) { Q_UNUSED(exp) return true; }
    /*!
     * \brief Do any needed initialization prior to connecting to hardware.
     * 
     * Derived classes must override this function, even if no action is performed.
     * 
     * \sa bcInitialize
     *
     */
    virtual void initialize() =0;

    /*!
     * \brief Attempt to communicate with hardware.
     * 
     * Derived classes must implement this function. An attempt should be made
     * to send a test command (e.g., `*IDN?`) to the device to ensure that the
     * correct device is connected and responsive. In the event of an error, a
     * message should be stored in ::d_errorString.
     * 
     * \sa bcTestConnection
     * 
     * \return Whether attempt was successful
     */
    virtual bool testConnection() =0;


private:
    /*!
     * \brief Perform read of auxiliary data
     * 
     * Derived classes may override this function to return data to be
     * displayed on the aux and/or rolling data plots.
     * 
     * \return AuxDataStorage::AuxDataMap (an alias for `std::map<QString,QVariant>`)
     * containing key-value pairs.
     */
    virtual AuxDataStorage::AuxDataMap readAuxData();
    
    /*!
     * \brief Perform read of validation data
     * 
     * Derived classes may override this function to return any additional data
     * that can be used for experiment validation which is **not** already
     * returned in readAuxData(). This may, for example, consist of digital
     * data that is not appropriate for graphical presentaiton.
     * 
     * \return AuxDataStorage::AuxDataMap (an alias for `std::map<QString,QVariant>`)
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
     * \brief Calls readAuxData() and emits auxDataRead()
     * 
     * \param event Timer event that caused the trigger.
     */
    void timerEvent(QTimerEvent *event) override;
};

#endif // HARDWAREOBJECT_H
