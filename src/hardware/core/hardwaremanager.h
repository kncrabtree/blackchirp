#ifndef HARDWAREMANAGER_H
#define HARDWAREMANAGER_H

#include <QObject>

#include <QList>
#include <QThread>

#include <src/data/datastructs.h>
#include <src/data/experiment/experiment.h>

class HardwareObject;
class FtmwScope;
class AWG;
class PulseGenerator;
class FlowController;
class IOBoard;
class MotorController;
class ClockManager;

#ifdef BC_PCONTROLLER
class PressureController;
#endif

#ifdef BC_TEMPCONTROLLER
class TemperatureController;
#endif

#ifdef BC_LIF
class LifScope;
class LifLaser;
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

    void ftmwScopeShotAcquired(const QByteArray);

    void clockFrequencyUpdate(BlackChirp::ClockType, double);
    void allClocksReady();

    void pGenSettingUpdate(int,BlackChirp::PulseSetting,QVariant);
    void pGenConfigUpdate(const PulseGenConfig);
    void pGenRepRateUpdate(double);

    void flowUpdate(int,double);
    void flowNameUpdate(int,QString);
    void flowSetpointUpdate(int,double);
    void gasPressureUpdate(double);
    void gasPressureSetpointUpdate(double);
    void gasPressureControlMode(bool);

#ifdef BC_PCONTROLLER
    void pressureControlReadOnly(bool);
    void pressureUpdate(double);
    void pressureSetpointUpdate(double);
    void pressureControlMode(bool);
#endif

#ifdef BC_LIF
    void lifScopeShotAcquired(LifTrace);
    void lifScopeConfigUpdated(BlackChirp::LifScopeConfig);
    void lifSettingsComplete(bool success = true);
    void lifLaserPosUpdate(double);
#endif

#ifdef BC_MOTOR
    void motorTraceAcquired(QVector<double> d);
    void motorMoveComplete(bool);
    void moveMotorToPosition(double x, double y, double z);
    void motorLimitStatus(BlackChirp::MotorAxis axis, bool negLimit, bool posLimit);
    void motorPosUpdate(BlackChirp::MotorAxis axis, double pos);
    void motorRest();
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

    void setClocks(const RfConfig rfc);

    void setPGenSetting(int index, BlackChirp::PulseSetting s, QVariant val);
    void setPGenConfig(const PulseGenConfig c);
    void setPGenRepRate(double r);

    void setFlowChannelName(int index, QString name);
    void setFlowSetpoint(int index, double val);
    void setGasPressureSetpoint(double val);
    void setGasPressureControlMode(bool en);

#ifdef BC_PCONTROLLER
    void setPressureSetpoint(double val);
    void setPressureControlMode(bool en);
    void openGateValve();
    void closeGateValve();
#endif

#ifdef BC_LIF
    void setLifParameters(double delay, double pos);
    bool setPGenLifDelay(double d);
    void setLifScopeConfig(const BlackChirp::LifScopeConfig c);
    bool setLifLaserPos(double pos);
#endif

private:
    int d_responseCount;
    void checkStatus();

    QList<HardwareObject*> d_hardwareList;
    FtmwScope *p_ftmwScope;
    AWG *p_awg;
    PulseGenerator *p_pGen;
    FlowController *p_flow;
    IOBoard *p_iob;
    ClockManager *p_clockManager;

#ifdef BC_PCONTROLLER
    PressureController *p_pc;
#endif

#ifdef BC_TEMPCONTROLLER
    TemperatureController *p_tc;
#endif

#ifdef BC_LIF
    LifScope *p_lifScope;
    LifLaser *p_lifLaser;
#endif

#ifdef BC_MOTOR
    MotorController *p_mc;
    MotorOscilloscope *p_motorScope;
#endif
};

#endif // HARDWAREMANAGER_H
