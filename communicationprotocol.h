#ifndef COMMUNICATIONPROTOCOL_H
#define COMMUNICATIONPROTOCOL_H

#include <QObject>

#include <QSettings>
#include <QApplication>

#include "datastructs.h"

class CommunicationProtocol : public QObject
{
    Q_OBJECT
public:
    enum CommType {
        Virtual,
        Tcp,
        Rs232,
        Gpib,
        Custom
    };

    explicit CommunicationProtocol(CommType type, QString key, QString subKey, QObject *parent = nullptr);
    virtual ~CommunicationProtocol();

    virtual bool writeCmd(QString cmd);
    virtual bool writeBinary(QByteArray dat);
    virtual QByteArray queryCmd(QString cmd);

    QString key() { return d_key; }
    CommType type() { return d_type; }
    QIODevice *device() { return p_device; }

    /*!
     * \brief Convenience function for setting read options
     * \param tmo Read timeout, in ms
     * \param useTermChar If true, look for termination characters at the end of a message
     * \param termChar Termination character(s)
     */
    void setReadOptions(int tmo, bool useTermChar = false, QByteArray termChar = QByteArray()) { d_timeOut = tmo, d_useTermChar = useTermChar, d_readTerminator = termChar; }

signals:
    void logMessage(QString,BlackChirp::LogMessageCode = BlackChirp::LogNormal);
    void hardwareFailure();

public slots:
    virtual void initialize() =0;
    virtual bool testConnection() =0;

protected:
    const CommType d_type;
    QString d_key;
    QString d_prettyName;

    QByteArray d_readTerminator; /*!< Termination characters that indicate a message from the device is complete. */
    bool d_useTermChar; /*!< If true, a read operation is complete when the message ends with d_readTerminator */
    int d_timeOut; /*!< Timeout for read operation, in ms */

    QIODevice *p_device = nullptr;
};

#endif // COMMUNICATIONPROTOCOL_H
