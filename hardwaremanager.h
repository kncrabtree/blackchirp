#ifndef HARDWAREMANAGER_H
#define HARDWAREMANAGER_H

#include <QObject>

#include <QList>
#include <QThread>

#include "datastructs.h"
#include "experiment.h"

class HardwareObject;
class FtmwScope;
class AWG;
class Synthesizer;
class PulseGenerator;
class FlowController;
class IOBoard;
class MotorController;

#ifdef BC_LIF
class LifScope;
#endif

#ifdef BC_MOTOR
class MotorController;
class MotorOscilloscope;
#endif

class HardwareManager : public QObject
{
    Q_OBJECT
public:
    explicit HardwareManager(QObject *parent = 0);
    ~HardwareManager();

signals:
    void logMessage(const QString, const BlackChirp::LogMessageCode = BlackChirp::LogNormal);
    void statusMessage(const QString);
    void hwInitializationComplete();

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
    void timeData(const QList<QPair<QString,QVariant>>,bool plot,QDateTime t = QDateTime::currentDateTime());
    void readTimeData();

    void ftmwScopeShotAcquired(const QByteArray);

    void valonTxFreqRead(double);
    void valonRxFreqRead(double);

    void pGenSettingUpdate(int,BlackChirp::PulseSetting,QVariant);
    void pGenConfigUpdate(const PulseGenConfig);
    void pGenRepRateUpdate(double);

    void flowUpdate(int,double);
    void flowNameUpdate(int,QString);
    void flowSetpointUpdate(int,double);
    void pressureUpdate(double);
    void pressureSetpointUpdate(double);
    void pressureControlMode(bool);

#ifdef BC_LIF
    void lifScopeShotAcquired(const LifTrace);
    void lifScopeConfigUpdated(const BlackChirp::LifScopeConfig);
    void lifSettingsComplete(bool success = true);
#endif

#ifdef BC_MOTOR
    void motorTraceAcquired(QVector<double> d);
    void motorMoveComplete(bool);
    void moveMotorToPosition(double x, double y, double z);
    void motorLimitStatus(BlackChirp::MotorAxis axis, bool negLimit, bool posLimit);
    void motorPosUpdate(BlackChirp::MotorAxis axis, double pos);
#endif

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
    void hardwareFailure();

    void sleep(bool b);

    void initializeExperiment(Experiment exp);

    void testAll();
    void testObjectConnection(const QString type, const QString key);

    void getTimeData();

    double setValonTxFreq(const double d);
    double setValonRxFreq(const double d);

    void setPGenSetting(int index, BlackChirp::PulseSetting s, QVariant val);
    void setPGenConfig(const PulseGenConfig c);
    void setPGenRepRate(double r);

    void setFlowChannelName(int index, QString name);
    void setFlowSetpoint(int index, double val);
    void setPressureSetpoint(double val);
    void setPressureControlMode(bool en);

#ifdef BC_LIF
    void setLifParameters(double delay, double frequency);
    bool setPGenLifDelay(double d);
    void setLifScopeConfig(const BlackChirp::LifScopeConfig c);
#endif

private:
    int d_responseCount;
    void checkStatus();

    QList<QPair<HardwareObject*,QThread*> > d_hardwareList;
    FtmwScope *p_ftmwScope;
    Synthesizer *p_synth;
    AWG *p_awg;
    PulseGenerator *p_pGen;
    FlowController *p_flow;
    IOBoard *p_iob;
#ifdef BC_LIF
    LifScope *p_lifScope;
#endif

#ifdef BC_MOTOR
    MotorController *p_mc;
    MotorOscilloscope *p_motorScope;
#endif
};

#endif // HARDWAREMANAGER_H
