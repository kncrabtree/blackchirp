#ifndef HARDWAREMANAGER_H
#define HARDWAREMANAGER_H

#include <QObject>
#include <QList>
#include <QThread>
#include "hardwareobject.h"
#include "loghandler.h"
#include "oscilloscope.h"
#include "experiment.h"

class HardwareManager : public QObject
{
    Q_OBJECT
public:
    explicit HardwareManager(QObject *parent = 0);
    ~HardwareManager();

signals:
    void logMessage(const QString, const LogHandler::MessageCode = LogHandler::Normal);
    void statusMessage(const QString);
    void allHardwareConnected(bool);
    /*!
     * \brief Emitted when a connection is being tested from the communication dialog
     * \param QString The HardwareObject key
     * \param bool Whether connection was successful
     * \param QString Error message
     */
    void testComplete(QString,bool,QString);
    void abortAcquisition();
    void experimentInitialized(Experiment);
    void scopeShotAcquired(const QByteArray);

public slots:
    void initialize();

    /*!
     * \brief Records whether hardware connection was successful
     * \param obj A HardwareObject that was tested
     * \param success Whether communication was sucessful
     * \param msg Error message
     */
    void connectionResult(HardwareObject *obj, bool success, QString msg);

    /*!
     * \brief Sets hardware status in d_status to false, disables program
     * \param obj The object that failed.
     *
     * TODO: Consider generating an abort signal here
     */
    void hardwareFailure(HardwareObject *obj, bool abort);

    void initializeExperiment(Experiment exp);

private:
    QHash<QString,bool> d_status;
    void checkStatus();

    QList<QPair<HardwareObject*,QThread*> > d_hardwareList;
    Oscilloscope *p_scope;

};

#endif // HARDWAREMANAGER_H
