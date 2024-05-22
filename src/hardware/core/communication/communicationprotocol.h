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
 * \ingroup ssKeys
 */
namespace BC::Key::Comm {
static const QString rs232{"rs232"};
static const QString tcp{"tcp"};
static const QString gpib{"gpib"};
static const QString custom{"custom"};
static const QString hwVirtual{"virtual"};
}

class CommunicationProtocol : public QObject
{
    Q_OBJECT
public:
    enum CommType {
        Virtual,
        Tcp,
        Rs232,
        Gpib,
        Custom,
        None
    };
    Q_ENUM(CommType)

    explicit CommunicationProtocol(QString key, QObject *parent = nullptr);
    virtual ~CommunicationProtocol();

    virtual bool writeCmd(QString cmd);
    virtual bool writeBinary(QByteArray dat);
    virtual QByteArray queryCmd(QString cmd, bool suppressError = false);

    const QString d_key;

    QIODevice *device() { return p_device; }
    QString errorString();

    /*!
     * \brief Convenience function for setting read options
     * \param tmo Read timeout, in ms
     * \param useTermChar If true, look for termination characters at the end of a message
     * \param termChar Termination character(s)
     */
    void setReadOptions(int tmo, bool useTermChar = false, QByteArray termChar = QByteArray()) { d_timeOut = tmo, d_useTermChar = useTermChar, d_readTerminator = termChar; }

signals:
    void logMessage(QString,LogHandler::MessageCode = LogHandler::Normal);
    void hardwareFailure();

public slots:
    bool bcTestConnection();
    virtual void initialize() =0;

protected:
    QString d_errorString;

    QByteArray d_readTerminator; /*!< Termination characters that indicate a message from the device is complete. */
    bool d_useTermChar; /*!< If true, a read operation is complete when the message ends with d_readTerminator */
    int d_timeOut; /*!< Timeout for read operation, in ms */

    virtual bool testConnection() =0;

    QIODevice *p_device = nullptr;
};

#endif // COMMUNICATIONPROTOCOL_H
