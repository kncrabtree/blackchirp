#ifndef HARDWAREMANAGER_H
#define HARDWAREMANAGER_H

#include <QObject>

#include <QList>
#include <QThread>

#include "loghandler.h"
#include "experiment.h"

class HardwareObject;
class FtmwScope;
class AWG;
class Synthesizer;
class PulseGenerator;
class FlowController;
class LifScope;

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
    void beginAcquisition();
    void abortAcquisition();
    void experimentInitialized(Experiment);
    void endAcquisition();
    void timeData(const QList<QPair<QString,QVariant>>);
    void readTimeData();

    void ftmwScopeShotAcquired(const QByteArray);

    void valonTxFreqRead(double);
    void valonRxFreqRead(double);

    void pGenSettingUpdate(int,PulseGenConfig::Setting,QVariant);
    void pGenConfigUpdate(const PulseGenConfig);
    void pGenRepRateUpdate(double);

    void flowUpdate(int,double);
    void flowNameUpdate(int,QString);
    void flowSetpointUpdate(int,double);
    void pressureUpdate(double);
    void pressureSetpointUpdate(double);
    void pressureControlMode(bool);

    void lifScopeShotAcquired(const LifConfig::LifScopeConfig, const QByteArray);

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

    void testObjectConnection(const QString type, const QString key);

    void getTimeData();

    double setValonTxFreq(const double d);
    double setValonRxFreq(const double d);

    void setPGenSetting(int index, PulseGenConfig::Setting s, QVariant val);
    void setPGenConfig(const PulseGenConfig c);
    void setPGenRepRate(double r);

    void setFlowChannelName(int index, QString name);
    void setFlowSetpoint(int index, double val);
    void setPressureSetpoint(double val);
    void setPressureControlMode(bool en);

private:
    QHash<QString,bool> d_status;
    void checkStatus();

    QList<QPair<HardwareObject*,QThread*> > d_hardwareList;
    FtmwScope *p_ftmwScope;
    Synthesizer *p_synth;
    AWG *p_awg;
    PulseGenerator *p_pGen;
    FlowController *p_flow;
    LifScope *p_lifScope;

};

#endif // HARDWAREMANAGER_H
