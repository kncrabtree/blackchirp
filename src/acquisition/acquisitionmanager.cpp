#include <acquisition/acquisitionmanager.h>

#include <math.h>
#include <QtConcurrent/QtConcurrent>
#include <QFutureWatcher>

AcquisitionManager::AcquisitionManager(QObject *parent) : QObject(parent), d_state(Idle)
{
}

AcquisitionManager::~AcquisitionManager()
{
}

void AcquisitionManager::beginExperiment(std::shared_ptr<Experiment> exp)
{
    ps_currentExperiment = exp;

    d_state = Acquiring;
    emit statusMessage(QString("Acquiring"));

    if(ps_currentExperiment->d_timeDataInterval > 0)
    {
        if(ps_currentExperiment->ftmwEnabled())
        {
            ps_currentExperiment->auxData()->registerKey(QString("Ftmw"),QString("Shots"));
            if(ps_currentExperiment->ftmwConfig()->d_phaseCorrectionEnabled)
            {
                ps_currentExperiment->auxData()->registerKey(QString("Ftmw"),QString("ChirpPhaseScore"));
                ps_currentExperiment->auxData()->registerKey(QString("Ftmw"),QString("ChirpShift"));
            }
            if(ps_currentExperiment->ftmwConfig()->d_chirpScoringEnabled)
                ps_currentExperiment->auxData()->registerKey(QString("Ftmw"),QString("ChirpRMS"));
        }

        auxDataTick();
        d_auxTimerId = startTimer(ps_currentExperiment->d_timeDataInterval*1000);
    }
    if(ps_currentExperiment->ftmwEnabled())
        emit newClockSettings(ps_currentExperiment->ftmwConfig()->d_rfConfig.getClocks());

    emit beginAcquisition();

#ifdef BC_LIF
    if(ps_currentExperiment->lifEnabled())
        emit nextLifPoint(ps_currentExperiment->lifConfig()->currentDelay(),
                      ps_currentExperiment->lifConfig()->currentLaserPos());
#endif
}

void AcquisitionManager::processFtmwScopeShot(const QByteArray b)
{
    if(d_state == Acquiring
            && ps_currentExperiment->ftmwEnabled()
            && !ps_currentExperiment->ftmwConfig()->isComplete()
            && !ps_currentExperiment->ftmwConfig()->d_processingPaused)
    {

        bool success = ps_currentExperiment->ftmwConfig()->addFids(b);
        auto errStr = ps_currentExperiment->ftmwConfig()->d_errorString;

        if(!success)
        {
            emit logMessage("Error processing FID data.",LogHandler::Error);
            abort();
            return;
        }
        else if(!errStr.isEmpty())
            emit logMessage(errStr,LogHandler::Warning);

        bool advanceSegment = ps_currentExperiment->ftmwConfig()->advance();

        if(advanceSegment)
            emit newClockSettings(ps_currentExperiment->ftmwConfig()->d_rfConfig.getClocks());

        emit ftmwUpdateProgress(ps_currentExperiment->ftmwConfig()->perMilComplete());
    }

    checkComplete();
}

#ifdef BC_LIF
void AcquisitionManager::processLifScopeShot(const QVector<qint8> b)
{
    if(d_state == Acquiring
            && ps_currentExperiment->lifEnabled()
            && !ps_currentExperiment->lifConfig()->d_processingPaused)
    {
        ps_currentExperiment->lifConfig()->addWaveform(b);
        emit lifPointUpdate();
        if(ps_currentExperiment->lifConfig()->advance() && !ps_currentExperiment->isComplete())
            emit nextLifPoint(ps_currentExperiment->lifConfig()->currentDelay(),
                              ps_currentExperiment->lifConfig()->currentLaserPos());

        emit lifShotAcquired(ps_currentExperiment->lifConfig()->perMilComplete());

    }

    checkComplete();
}

void AcquisitionManager::lifHardwareReady(bool success)
{
    if(ps_currentExperiment.get())
    {
        if(ps_currentExperiment->lifEnabled())
        {
            if(!success)
            {
                emit logMessage(QString("LIF delay and/or frequency could not be set. Aborting."),LogHandler::Error);
                abort();
            }
            else
                ps_currentExperiment->lifConfig()->hwReady();
        }
    }
}
#endif

void AcquisitionManager::processAuxData(AuxDataStorage::AuxDataMap m)
{
	if(d_state == Acquiring)
	{
        emit auxData(m,ps_currentExperiment->auxData()->currentPointTime());
        if(!ps_currentExperiment->addAuxData(m))
            abort();
    }
}

void AcquisitionManager::processValidationData(AuxDataStorage::AuxDataMap m)
{
    if(d_state == Acquiring)
    {
        for(auto &[key,val] : m)
        {
            if(!ps_currentExperiment->validateItem(key,val))
            {
                abort();
                break;
            }
        }
    }
}

void AcquisitionManager::clockSettingsComplete(const QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks)
{
    if(d_state == Acquiring && ps_currentExperiment->ftmwEnabled())
    {
        ps_currentExperiment->ftmwConfig()->d_rfConfig.setCurrentClocks(clocks);
        ps_currentExperiment->ftmwConfig()->hwReady();
    }
}

void AcquisitionManager::pause()
{
    if(d_state == Acquiring)
    {
        d_state = Paused;
        emit statusMessage(QString("Paused"));
    }
}

void AcquisitionManager::resume()
{
    if(d_state == Paused)
    {
        d_state = Acquiring;
        emit statusMessage(QString("Acquiring"));
    }
}

void AcquisitionManager::abort()
{
    if(d_state == Paused || d_state == Acquiring)
    {
        ps_currentExperiment->abort();
        finishAcquisition();
    }
}

void AcquisitionManager::auxDataTick()
{
    ps_currentExperiment->auxData()->startNewPoint();
    if(ps_currentExperiment->ftmwEnabled())
    {
        AuxDataStorage::AuxDataMap m;
        m.emplace(AuxDataStorage::makeKey("Ftmw","Shots"),
                  ps_currentExperiment->ftmwConfig()->completedShots());
        if(ps_currentExperiment->ftmwConfig()->d_chirpScoringEnabled)
            m.emplace(AuxDataStorage::makeKey("Ftmw","ChirpRMS"),
                      ps_currentExperiment->ftmwConfig()->chirpRMS());
        if(ps_currentExperiment->ftmwConfig()->d_phaseCorrectionEnabled)
        {
            m.emplace(AuxDataStorage::makeKey("Ftmw","ChirpPhaseScore"),
                      ps_currentExperiment->ftmwConfig()->chirpFOM());
            m.emplace(AuxDataStorage::makeKey("Ftmw","ChirpShift"),
                      ps_currentExperiment->ftmwConfig()->chirpShift());
        }

        processAuxData(m);

    }
    emit auxDataSignal();
}

void AcquisitionManager::checkComplete()
{
    if(d_state == Acquiring)
    {
        if(ps_currentExperiment->canBackup())
        {
            QFutureWatcher<void> fw;
            connect(&fw,&QFutureWatcher<void>::finished,this,&AcquisitionManager::backupComplete);
            fw.setFuture(QtConcurrent::run([this]{ ps_currentExperiment->backup(); }));
        }
        if(ps_currentExperiment->isComplete())
            finishAcquisition();
    }
}

void AcquisitionManager::finishAcquisition()
{
    emit endAcquisition();
    d_state = Idle;


    if(!ps_currentExperiment->isDummy())
    {
        emit statusMessage(QString("Saving experiment %1").arg(ps_currentExperiment->d_number));
        ps_currentExperiment->finalSave();
    }

    emit experimentComplete();
    ps_currentExperiment.reset();
}

void AcquisitionManager::timerEvent(QTimerEvent *event)
{
    if(d_state == Acquiring && event->timerId() == d_auxTimerId)
    {
        auxDataTick();
        event->accept();
        return;
    }

    QObject::timerEvent(event);

}
