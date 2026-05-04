#ifndef COMMUNICATIONPROTOCOL_H
#define COMMUNICATIONPROTOCOL_H

#include <QObject>

#include <QSettings>
#include <QApplication>

#include <data/loghandler.h>
#include <data/storage/settingsstorage.h>

/*!
 * CommunicationProtocol keys for SettingsStorage
 * 
 * These keys are used by the HardwareManager to create groups of settings for
 * the CommunicationDialog.
 * 
 */
namespace BC::Key::Comm {
static const QString rs232{"rs232"}; /*!< Rs232Instrument */
static const QString tcp{"tcp"}; /*!< TcpInstrument */
static const QString gpib{"gpib"}; /*!< GpibInstrument */
static const QString custom{"custom"}; /*!< CustomInstrument */
static const QString hwVirtual{"virtual"}; /*!< VirtaulInstrument */
}

/*!
 * \brief Represents a communication interface for a HardwareObject
 * 
 * The CommunicationProtocol is a light wrapper around a QIODevice (if
 * appropriate) which handles the actual communication. Extra features are
 * available for customizing the communication and executing common patterns
 * (queries, timeouts, parsing replies with termination characters, etc).
 * 
 * CommunicationProtocol is an interface class; subclasses must implement the
 * _device() function which returns the underlying QIODevice (which may safely
 * be nullptr if no such device exists). The device itself should be created in
 * the initialize() function, which must also be implemented in a subclass. For
 * instance, Rs232Instrument and TcpInstrument both return pointers to their
 * underlying QSerialPort and QTcpSocket devices, while the other
 * implementations return nullptr. The _device() function itself may be called
 * externally if direct access to the QIODevice is needed (i.e., if the
 * QIODevice functionality is not exposed by the wrapper), and the device()
 * template function can be called to conveniently cast to a derived type. For
 * example, if the device is a QTcpSocket:
 * 
 *     CommunicationProtocol *comm = new TcpInstrument("key");
 *     comm->initialize();
 *     auto socket = comm->device<QTcpSocket>();
 *     //socket is now a QTcpSocket*
 *     
 *     auto socket2 = comm->device<QSerialPort>();
 *     //socket2 is nullptr because comm is a TcpInstrument
 *     
 * CommunicationProtocol provides 3 convenience functions with default
 * implementations that may be extended or overwritten by subclasses:
 * 
 *   - writeCmd() writes an ASCII string to the QIODevice
 *   - writeBinary() writes binary data to the QIODevice
 *   - queryCmd() writes a command to the device and reads a response.
 *   
 * The read behavior can be set by a call to setReadOptions, which allows for
 * specifying a query timeout and query termination character(s) that are used
 * to detect the end of a message.
 * 
 */
class CommunicationProtocol : public QObject
{
    Q_OBJECT
public:
    /*!
     * \brief CommunicationProtocol options
     * 
     * Used by HardwareObject to determine which subclass to create.
     * 
     */
    enum CommType {
        Virtual,
        Tcp,
        Rs232,
        Gpib,
        Custom,
        None
    };
    Q_ENUM(CommType)

    /*!
     * \brief Constructor
     * \param key Key assigned to ::d_key
     * \param parent QObject parent
     */
    explicit CommunicationProtocol(QString key, QObject *parent = nullptr);
    
    /*!
     * \brief Destructor. Does nothing
     */
    virtual ~CommunicationProtocol();

    /*!
     * \brief Writes `cmd` to the device as ASCII-encoded data
     * 
     * This function can be safely called even if the device is nullptr.
     * 
     * \param cmd Data to write
     * \return Whether data was written successfully
     */
    virtual bool writeCmd(QString cmd);
    
    /*!
     * \brief Writes `dat` to the device as binary data
     * 
     * This function can be safely called even if the device is nullptr
     * 
     * \param dat Data to write
     * \return Whether data was written successfully
     */
    virtual bool writeBinary(QByteArray dat);
    
    /*!
     * \brief Writes `cmd` to the device and attempts to read a response
     * 
     * This function can be safely called even if the device is nullptr. The
     * settings used in the read portion of this function should be set using
     * setReadOptions(). By default, a hardwareFailure() signal is emitted if
     * there is a problem reading or writing data. This signal can be blocked
     * by setting `suppressError` to true. This may be useful if intermiitent
     * failures are expected and are handled by the caller.
     * 
     * \param cmd
     * \param suppressError
     * \return Repsonse from device
     */
    virtual QByteArray queryCmd(QString cmd, bool suppressError = false);

    /*!
     * \brief Reads n bytes from device, respecting timeout
     * \param n Number of bytes to read
     * \param suppressError If true, no failure signal will be emitted if all bytes are not read
     * \return Bytes read
     */
    virtual QByteArray readBytes(qint64 n, bool suppressError = false);

    const QString d_key; /*!< Key used to identify communication protocol. Not currently used. */

    /*!
     * \brief Returns a pointer to the underlying device.
     * 
     * \return QIODevice pointer or nullptr
     */
    virtual QIODevice *_device() =0;
    
    /*!
     * \brief Convenience function for casting QIODevice to type T
     * \return Pointer to device of type T* or nullptr
     */
    template<typename T>
    T* device() { return dynamic_cast<T*>(_device()); }
    
    /*!
     * \brief Sets contents of error string
     * \param Error message
     */
    void setErrorString(const QString str);
    
    /*!
     * \brief Returns last error and clears the error string.
     * \return Error message
     */
    QString errorString();

    /*!
     * \brief Convenience function for setting read options
     * \param tmo Read timeout, in ms
     * \param useTermChar Whether to look for termination characters at the end of a message
     * \param termChar Termination character(s)
     */
    void setReadOptions(int tmo, bool useTermChar = false, QByteArray termChar = QByteArray()) { d_timeOut = tmo, d_useTermChar = useTermChar, d_readTerminator = termChar; }

signals:
    /*!
     * \brief Sends message to Log for printing
     * \param QString Message
     * \param LogHandler::MessageCode Type of message
     */
    void logMessage(QString,LogHandler::MessageCode = LogHandler::Normal);
    
    /*!
     * \brief Emitted when a failure occurs with the QIODevice
     * 
     * \sa HardwareObject::hardwareFailure()
     */
    void hardwareFailure();

public slots:
    /*!
     * \brief Attempts to open the QIODevice
     * 
     * This function calls testConnection() and grabs the error string if the
     * call fails.
     * 
     * \return Whether the QIODevice was opened successfully
     */
    bool bcTestConnection();
    
    /*!
     * \brief Creates the QIODevice
     */
    virtual void initialize() =0;

private:
    QString d_errorString; /*!< Most recent error message */

    QByteArray d_readTerminator; /*!< Termination characters that indicate a message from the device is complete. */
    bool d_useTermChar{false}; /*!< If true, a read operation is complete when the message ends with d_readTerminator */
    int d_timeOut{1000}; /*!< Timeout for read operation, in ms */

    /*!
     * \brief Attempts to open the QIODevice
     * 
     * This function should return true if the device was opened successfully
     * or if there is no QIODevice to open.
     * 
     * \return Whether device was opened successfully.
     */
    virtual bool testConnection() =0;

};

#endif // COMMUNICATIONPROTOCOL_H
