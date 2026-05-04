#ifndef COMMUNICATIONPROTOCOL_H
#define COMMUNICATIONPROTOCOL_H

#include <QObject>

#include <QSettings>
#include <QCoreApplication>

#include <data/loghandler.h>
#include <data/storage/settingsstorage.h>


/*!
 * \brief Communication interface for a HardwareObject.
 *
 * Interface class. Subclasses provide a transport (e.g. Rs232Instrument
 * wraps QSerialPort, TcpInstrument wraps QTcpSocket; VirtualInstrument
 * and CustomInstrument keep the device pointer null when no
 * QIODevice representation is appropriate) by overriding _device() to
 * return the underlying QIODevice. The device itself is created in
 * initialize(), which subclasses must also implement. Read behavior —
 * query timeout and termination characters — is loaded from settings
 * by loadCommReadOptions().
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
    virtual bool writeCmd(const QString &cmd);
    
    /*!
     * \brief Writes `dat` to the device as binary data
     * 
     * This function can be safely called even if the device is nullptr
     * 
     * \param dat Data to write
     * \return Whether data was written successfully
     */
    virtual bool writeBinary(const QByteArray &dat);
    
    /*!
     * \brief Writes `cmd` to the device and attempts to read a response
     * 
     * This function can be safely called even if the device is nullptr. The
     * settings used in the read portion of this function should be set using
     * setReadOptions(). By default, a hardwareFailure() signal is emitted if
     * there is a problem reading or writing data. This signal can be blocked
     * by setting `suppressError` to true. This may be useful if intermittent
     * failures are expected and are handled by the caller.
     *
     * \param cmd Command to send to the device
     * \param suppressError If true, suppress the hardwareFailure() signal on read/write error
     * \return Response from device
     */
    virtual QByteArray queryCmd(const QString &cmd, bool suppressError = false);

    /*!
     * \brief Reads n bytes from device, respecting timeout
     * \param n Number of bytes to read
     * \param suppressError If true, no failure signal will be emitted if all bytes are not read
     * \return Bytes read
     */
    virtual QByteArray readBytes(qint64 n, bool suppressError = false);

    const QString d_key; /*!< Key used to identify the communication protocol instance. */

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
     * \param str Error message
     */
    void setErrorString(const QString str);
    
    /*!
     * \brief Returns last error and clears the error string.
     * \return Error message
     */
    QString errorString();

    /*!
     * \brief Loads read options from settings for this communication protocol
     */
    void loadCommReadOptions();

signals:
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
     * \brief Sets read options for communication protocol
     * \param timeout Read timeout in ms (<=0 disables timeout)
     * \param termChar Termination character(s) (empty disables termChar)
     */
    void setReadOptions(int timeout, const QString& termChar) { 
        d_timeOut = timeout; 
        d_readTerminator = termChar.toUtf8();
        d_useTermChar = !termChar.isEmpty();
    }

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
