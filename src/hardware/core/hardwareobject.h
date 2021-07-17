#ifndef HARDWAREOBJECT_H
#define HARDWAREOBJECT_H

#include <QObject>
#include <QString>
#include <QSettings>
#include <QApplication>
#include <QList>
#include <QPair>

#include <data/datastructs.h>
#include <data/storage/settingsstorage.h>
#include <data/storage/auxdatastorage.h>
#include <hardware/core/communication/communicationprotocol.h>
#include <hardware/core/communication/virtualinstrument.h>
#include <hardware/core/communication/tcpinstrument.h>
#include <hardware/core/communication/rs232instrument.h>
#include <hardware/core/communication/custominstrument.h>

#ifdef BC_GPIBCONTROLLER
#include <hardware/core/communication/gpibinstrument.h>
#endif

#include <data/experiment/experiment.h>

namespace BC::Key::HW {
static const QString key("key");
static const QString subKey("subKey");
static const QString name("prettyName");
static const QString connected("connected");
static const QString critical("critical");
static const QString threaded("threaded");
static const QString commType("commType");
static const QString rInterval("rollingDataIntervalSec");
}

/*!
 * \brief Abstract base class for all hardware connected to the instrument.
 *
 * This class establishes a common interface for all hardware.
 * Adding a new piece of hardware to the code involves creating a subclass of HardwareObject (or of one of the existing subclasses: see TcpInstrument, GpibInstrument, and Rs232Instrument)
 * Each hardware object has a name (d_prettyName and name()) that is used in the user interface, and a key (d_key and key()) that is used to refer to the object in the settings file.
 * Subclasses must assign strings to these variables in their constructors.
 * For communication with the UI, a logMessage() signal exists that can print a message in the log with an optional status code.
 * In general, this should only be used for reporting errors, but it is also useful in debugging.
 *
 * HardwareObjects are designed to be used in their own threads for the most part.
 * Because of this, the constructor should generally not be called with a parent, and any QObjects that will be created in the child class with this as the parent should NOT be initialized in the constructor.
 * Generally, a HardwareObject is created, signal-slot connections are made, and then the object is pushed into its own thread.
 * When the thread starts, the bcInitInstrument function is called, which reads settings, calls the initialize() pure virtual function, then calls bcTestConnection().
 * Any functions that need to be called from outside the class (e.g., from the HardwareManager during a scan) should be declared as slots, and QMetaObject::invokeMethod used to activate them (optionally with the Qt::BlockingQueuedConnection flag set if it returns a value or other actions need to wait until it is complete).
 * If for some reason this is not possible, then data members need to be protected by a QMutex to prevent concurrent access.
 * In the initialize function, any QObjects can be created and other settings made, because the object is already in its new thread.
 *
 * The testConnection() protected function should attempt to establish communication with the physical device and perform some sort of test to make sure that the connection works and that the correct device is targeted.
 * Usually, this takes the form of an ID query of some type (e.g., *IDN?).
 * If unsuccessful, a message should be stored in d_errorString.
 * The bcTestConnection() function wraps around the testConnection() function and handles communicating with the rest of the program.
 * The bcTestConnection() slot is also called from the communication dialog to re-test a connection that is lost.
 *
 * For instruments that do not communicate by GPIB, RS232, or TCP (i.e., a CustomInstrument), a general interface is available to allow the user to specify any information needed to connect to the device (file handles, ID numbers, etc).
 * In the constructor of the implementation, create a QSettings Array with the key: d_key\d_subKey\comm.
 * Each entry may define the following parameters:
 * name: A QString that will be displayed to the user to indicate what information is sought (e.g., Device name, ID number, etc)
 * key: A QString that the implementation will use to retrieve the entered value
 * type: "string" or "int" -- String will create a QLineEdit object for text entry, and "int" will create a QSpinBox for numeric entry
 * length: (optional) sets a limit on the length of text entry for strings
 * min: (optional) sets the lower limit on numeric entries (defaut: -2^31)
 * max: (optional) sets the uppser limit on numeric entries (default: 2^31-1)
 *
 * If at any point during operation, the program loses communication with the device, the hardwareFailure() signal should be emitted with the this pointer.
 * When emitted, any ongoing scan will be terminated and all controls will be frozen until communication is re-established by a successful call to the testConnection() function.
 * Some communication options (timeout, termination chartacters, etc) can be set, but it is up to the subclass to decide how or whether to make use of them.
 * They are provided here for convenience because TCP Instruments and RS232 instruments both use them.
 *
 * Finally, the sleep() function may be implemented in a subclass if anything needs to be done to put the device into an inactive state.
 * For instance, the FlowController turns off gas flows to conserve sample, and the PulseGenerator turns off all pulses.
 * If sleep() is implemented, it is recommneded to explicitly call HardwareObject::sleep(), as this will display a message in the log stating that the device is in fact asleep.
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
    explicit HardwareObject(const QString key, const QString subKey, const QString name,
                            CommunicationProtocol::CommType commType, QObject *parent = nullptr,
                            bool threaded = true, bool critical = true);
    virtual ~HardwareObject();

    const QString d_name; /*!< Name to be displayed on UI */
    const QString d_key; /*!< Name to be used in settings for abstract hardware*/
    const QString d_subKey; /*< Name to be used in settings for real hardware*/

    const bool d_critical;
    const bool d_threaded;
    const CommunicationProtocol::CommType d_commType;

    QString errorString();

    bool isConnected() const { return d_isConnected; }

    virtual QStringList validationKeys() const { return {}; }
	
signals:
    /*!
     * \brief Displays a message on the log.
     * \param QString The message to display
     * \param BlackChirp::MessageCode The status incidator (Normal, Warning, Error, Highlight)
     */
	void logMessage(const QString, const BlackChirp::LogMessageCode = BlackChirp::LogNormal);

    /*!
     * \brief Indicates whether a connection is successful
     * \param HardwareObject* This pointer
     * \param bool True if connection is successful
     * \param msg If unsuccessful, an optional message to display on the log tab along with the failure message.
     */
    void connected(bool success,QString msg,QPrivateSignal);

    /*!
     * \brief Signal emitted if communication to hardware fails.
     */
    void hardwareFailure();

    void auxDataRead(AuxDataStorage::AuxDataMap,QPrivateSignal);
    void validationDataRead(AuxDataStorage::AuxDataMap,QPrivateSignal);
    void rollingDataRead(AuxDataStorage::AuxDataMap,QPrivateSignal);
	
public slots:
    void bcInitInstrument();
    void bcTestConnection();
    void bcReadAuxData();
    void setRollingTimerInterval(int interval);

    virtual void readSettings();

    /*!
     * \brief Puts device into a standby mode.
     * \param b If true, go into standby mode. Else, active mode.
     */
	virtual void sleep(bool b);

    virtual bool prepareForExperiment(Experiment &exp) { Q_UNUSED(exp) return true; }
    virtual void beginAcquisition(){}
    virtual void endAcquisition(){}

    virtual void buildCommunication(QObject *gc = nullptr);

protected:
    QString d_errorString;
    bool d_enabledForExperiment;
    CommunicationProtocol *p_comm;

    /*!
     * \brief Do any needed initialization prior to connecting to hardware. Pure virtual
     *
     */
    virtual void initialize() =0;

    /*!
     * \brief Attempt to communicate with hardware. Pure virtual.
     * \return Whether attempt was successful
     */
    virtual bool testConnection() =0;


private:
    virtual AuxDataStorage::AuxDataMap readAuxData();
    virtual AuxDataStorage::AuxDataMap readValidationData();

    bool d_isConnected;
    int d_rollingDataTimerId{-1};


	

    // QObject interface
protected:
    void timerEvent(QTimerEvent *event) override;
};

#endif // HARDWAREOBJECT_H
