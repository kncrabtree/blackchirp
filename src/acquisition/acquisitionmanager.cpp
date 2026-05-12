#include <acquisition/acquisitionmanager.h>

#include <data/experiment/ftmwconfigtypes.h>
#include <data/storage/fidstoragebase.h>
#include <data/storage/waveformbuffer.h>
#include <data/crashhandler.h>

#include <math.h>

AcquisitionManager::AcquisitionManager(QObject *parent) : QObject(parent), d_state(Idle)
{
    pu_fw = std::make_unique<QFutureWatcher<void>>();
    connect(pu_fw.get(),&QFutureWatcher<void>::finished,this,&AcquisitionManager::onBackupFinished);
}

AcquisitionManager::~AcquisitionManager()
{
    d_abortProcessing.store(true, std::memory_order_release);
    if(pu_processingWatcher && pu_processingWatcher->isRunning())
        pu_processingWatcher->waitForFinished();
}

void AcquisitionManager::beginExperiment(std::shared_ptr<Experiment> exp)
{
    ps_currentExperiment = exp;
    CrashHandler::setActiveExperiment(exp ? exp->d_number : 0);

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

    if(ps_currentExperiment->ftmwEnabled() && ps_currentExperiment->ftmwConfig()->waveformBuffer())
    {
        d_abortProcessing.store(false, std::memory_order_relaxed);

        pu_processingWatcher = std::make_unique<QFutureWatcher<FtmwProcessingResult>>();
        connect(pu_processingWatcher.get(), &QFutureWatcher<FtmwProcessingResult>::finished,
                this, &AcquisitionManager::onProcessingComplete);

        p_drainTimer = new QTimer(this);
        p_drainTimer->setInterval(20);
        connect(p_drainTimer, &QTimer::timeout, this, &AcquisitionManager::drainFtmwBuffer);
        p_drainTimer->start();
    }

    if(ps_currentExperiment->lifEnabled())
        emit nextLifPoint(ps_currentExperiment->lifConfig()->currentDelay(),
                      ps_currentExperiment->lifConfig()->currentLaserPos());
}

void AcquisitionManager::drainFtmwBuffer()
{
    if(d_state != Acquiring || !ps_currentExperiment->ftmwEnabled())
        return;

    auto *ftmw = ps_currentExperiment->ftmwConfig();
    auto *buf = ftmw->waveformBuffer();
    if(!buf || buf->available() == 0)
        return;

    // Don't start new processing if worker is still running
    if(pu_processingWatcher && pu_processingWatcher->isRunning())
        return;

    // Read entries from buffer (fast — just moves QByteArrays)
    std::vector<WaveformEntry> entries;
    WaveformEntry entry;

    while(buf->read(entry))
    {
        if(entry.flushMarker)
            break;

        if(ftmw->isComplete() || ftmw->d_processingPaused)
            continue;

        entries.push_back(std::move(entry));
    }

    if(entries.empty())
    {
        // Only flush markers or all entries skipped — still call advance
        // for segment boundary and autosave handling
        bool advanceSegment = ftmw->advance();
        if(advanceSegment)
            emit newClockSettings(ftmw->d_rfConfig.getClocks());
        checkComplete();
        return;
    }

    // Pause drain timer while the worker processes
    p_drainTimer->stop();

    // Dispatch to worker thread for expensive parse + accumulate
    pu_processingWatcher->setFuture(
        QtConcurrent::run([this, ftmw, data = std::move(entries)]() mutable -> FtmwProcessingResult {
            FtmwProcessingResult result;

            if(d_abortProcessing.load(std::memory_order_acquire))
                return result;

#ifndef BC_CUDA
            bool success = ftmw->addBatchFids(data);
            if(!success)
            {
                result.success = false;
                result.errorString = ftmw->d_errorString;
            }
            else if(!ftmw->d_errorString.isEmpty())
                result.warningString = ftmw->d_errorString;
            result.entriesProcessed = static_cast<int>(data.size());
#else
            for(auto &e : data)
            {
                if(d_abortProcessing.load(std::memory_order_acquire))
                    break;

                bool success;
                if(e.preAccumulated)
                    success = ftmw->addPreAccumulatedFids(e.data, e.shotCount);
                else
                    success = ftmw->addFids(e.data);

                result.entriesProcessed++;

                if(!success)
                {
                    result.success = false;
                    result.errorString = ftmw->d_errorString;
                    break;
                }

                if(!ftmw->d_errorString.isEmpty() && result.warningString.isEmpty())
                    result.warningString = ftmw->d_errorString;
            }
#endif

            return result;
        })
    );
}

void AcquisitionManager::onProcessingComplete()
{
    // Guard: if acquisition has already ended (e.g., abort called
    // finishAcquisition while the worker was running), do nothing.
    if(d_state == Idle || !ps_currentExperiment)
        return;

    auto result = pu_processingWatcher->result();

    if(d_abortProcessing.load(std::memory_order_acquire))
        return;

    if(!result.success)
    {
        bcError("Error processing FID data."_L1);
        abort();
        return;
    }

    if(!result.warningString.isEmpty())
        bcWarn(result.warningString);

    auto *ftmw = ps_currentExperiment->ftmwConfig();

    bool advanceSegment = ftmw->advance();
    if(advanceSegment)
        emit newClockSettings(ftmw->d_rfConfig.getClocks());

    emit ftmwUpdateProgress(ftmw->perMilComplete());
    checkComplete();

    // Restart drain timer if still acquiring
    if(d_state == Acquiring && p_drainTimer)
        p_drainTimer->start();
}

void AcquisitionManager::processLifDigitizerShot(const QVector<qint8> b)
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
                bcError("LIF delay and/or frequency could not be set. Aborting."_L1);
                abort();
            }
            else
                ps_currentExperiment->lifConfig()->hwReady();
        }
    }
}

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
            d_lastBackupWasManual = false;
            dispatchBackup();
        }
        if(ps_currentExperiment->isComplete())
            finishAcquisition();
    }
}

bool AcquisitionManager::ftmwModeSupportsBackup() const
{
    if(!ps_currentExperiment || !ps_currentExperiment->ftmwEnabled())
        return false;
    const auto t = ps_currentExperiment->ftmwConfig()->d_type;
    return t == FtmwConfig::Target_Shots
            || t == FtmwConfig::Target_Duration
            || t == FtmwConfig::Forever;
}

void AcquisitionManager::dispatchBackup()
{
    // Stamp d_lastBackupTime on this thread before handing the file write
    // to the thread pool. The periodic backup check reads d_lastBackupTime
    // on this thread, so the write/read pair stays thread-confined.
    ps_currentExperiment->d_lastBackupTime = QDateTime::currentDateTime();
    pu_fw->setFuture(QtConcurrent::run([this]{ ps_currentExperiment->backup(); }));
}

void AcquisitionManager::requestBackup()
{
    if(d_state == Idle || !ps_currentExperiment || !ftmwModeSupportsBackup())
        return;

    // Click-time confirmation. The completion message replaces this.
    emit statusMessage(QString("Manual backup requested"),3000);

    if(pu_fw && pu_fw->isRunning())
    {
        // An automatic backup is already saving the current FID list — the
        // user's manual request is satisfied by that same write. Promote the
        // in-flight backup so onBackupFinished() reports it as manual.
        d_lastBackupWasManual = true;
        return;
    }

    d_lastBackupWasManual = true;
    dispatchBackup();
}

void AcquisitionManager::onBackupFinished()
{
    if(d_lastBackupWasManual)
    {
        int n = 0;
        if(ps_currentExperiment && ps_currentExperiment->ftmwEnabled())
            n = ps_currentExperiment->ftmwConfig()->storage()->numBackups();
        const auto msg = QString("Manual backup complete (backup %1)").arg(n);
        bcHighlight(msg);
        emit statusMessage(msg,5000);

        // Restore the persistent acquisition-state label after the timed
        // status message expires, so the bar does not stay blank.
        QTimer::singleShot(5000,this,[this]{
            if(d_state == Acquiring)
                emit statusMessage(QString("Acquiring"));
            else if(d_state == Paused)
                emit statusMessage(QString("Paused"));
        });
    }
    d_lastBackupWasManual = false;
    emit backupComplete();
}

void AcquisitionManager::finishAcquisition()
{
    if(p_drainTimer)
    {
        p_drainTimer->stop();
        delete p_drainTimer;
        p_drainTimer = nullptr;
    }

    // Signal worker to exit early and wait for it to finish.
    // Worst-case latency: one addFids call (~300-800ms for very large waveforms).
    d_abortProcessing.store(true, std::memory_order_release);
    if(pu_processingWatcher)
    {
        if(pu_processingWatcher->isRunning())
            pu_processingWatcher->waitForFinished();
        pu_processingWatcher->disconnect();
        pu_processingWatcher.reset();
    }

    emit endAcquisition();
    d_state = Idle;


    if(!ps_currentExperiment->isDummy())
    {
        emit statusMessage(QString("Saving experiment %1").arg(ps_currentExperiment->d_number));
        ps_currentExperiment->finalSave();
    }

    emit experimentComplete();
    ps_currentExperiment.reset();
    CrashHandler::setActiveExperiment(0);
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
