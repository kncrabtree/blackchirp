#include <acquisition/acquisitionmanager.h>

#include <math.h>
#include <QtConcurrent/QtConcurrent>
#include <QFutureWatcher>

AcquisitionManager::AcquisitionManager(QObject *parent) : QObject(parent), d_state(Idle), d_currentShift(0), d_lastFom(0.0)
{
#ifdef BC_MOTOR
    d_waitingForMotor = false;
#endif
}

AcquisitionManager::~AcquisitionManager()
{
}

void AcquisitionManager::beginExperiment(std::shared_ptr<Experiment> exp)
{

    d_currentShift = 0;
    d_lastFom = 0.0;
    ps_currentExperiment = exp;

    d_state = Acquiring;
    emit statusMessage(QString("Acquiring"));

    if(ps_currentExperiment->d_timeDataInterval > 0)
    {
        if(ps_currentExperiment->ftmwEnabled())
            ps_currentExperiment->auxData()->registerKey(QString("Ftmw"),QString("Shots"));

        auxDataTick();
        d_auxTimerId = startTimer(ps_currentExperiment->d_timeDataInterval*1000);
    }
    emit beginAcquisition();

#ifdef BC_MOTOR
    if(d_currentExperiment->motorScan().isEnabled())
    {
        d_waitingForMotor = true;
        QVector3D pos = d_currentExperiment->motorScan().currentPos();
        emit startMotorMove(pos.x(),pos.y(),pos.z());
        emit statusMessage(QString("Moving motor to (X,Y,Z) = (%1, %2, %3)")
                           .arg(pos.x(),0,'f',3).arg(pos.y(),0,'f',3).arg(pos.z(),0,'f',3));
    }
#endif

}

void AcquisitionManager::processFtmwScopeShot(const QByteArray b)
{
    if(d_state == Acquiring
            && ps_currentExperiment->ftmwEnabled()
            && !ps_currentExperiment->ftmwConfig()->isComplete()
            && !ps_currentExperiment->ftmwConfig()->processingPaused())
    {

        bool success = true;

        if(ps_currentExperiment->ftmwConfig()->d_chirpScoringEnabled)
        {
            success = scoreChirp(b);
            if(!success)
                return;
        }

        if(ps_currentExperiment->ftmwConfig()->d_phaseCorrectionEnabled)
        {
            success = calculateShift(b);
            if(!success)
                return;
        }

        success = ps_currentExperiment->ftmwConfig()->addFids(b,d_currentShift);

        if(!success)
        {
            emit logMessage(ps_currentExperiment->d_errorString,LogHandler::Error);
            abort();
            return;
        }

        bool advanceSegment = ps_currentExperiment->ftmwConfig()->advance();

        if(advanceSegment)
        {
#ifdef BC_CUDA
#pragma message("Move to FTMWconfig")
//            gpuAvg.setCurrentData(d_currentExperiment.ftmwConfig()->rawFidList());
#endif
            emit newClockSettings(ps_currentExperiment->ftmwConfig()->d_rfConfig.getClocks());
        }

        emit ftmwUpdateProgress(ps_currentExperiment->ftmwConfig()->perMilComplete());
    }

    checkComplete();
}

#ifdef BC_LIF
void AcquisitionManager::processLifScopeShot(const LifTrace t)
{
    if(d_state == Acquiring && d_currentExperiment.lifConfig().isEnabled() && !d_currentExperiment.isLifWaiting())
    {
        //process trace; only send data to UI if point is complete
        if(d_currentExperiment.addLifWaveform(t))
        {
            emit lifPointUpdate(d_currentExperiment.lifConfig());

            if(!d_currentExperiment.isComplete())
            {
                d_currentExperiment.setLifWaiting(true);
                emit nextLifPoint(d_currentExperiment.lifConfig().currentDelay(),
                                  d_currentExperiment.lifConfig().currentLaserPos());
            }
        }

        if(d_currentExperiment.lifConfig().completedShots() <= d_currentExperiment.lifConfig().totalShots())
            emit lifShotAcquired(d_currentExperiment.lifConfig().completedShots());

    }

    checkComplete();
}

void AcquisitionManager::lifHardwareReady(bool success)
{
    if(d_currentExperiment.lifConfig().isEnabled())
    {
        if(!success)
        {
            emit logMessage(QString("LIF delay and/or frequency could not be set. Aborting."),LogHandler::Error);
            abort();
        }
        else
            d_currentExperiment.setLifWaiting(false);
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
        //save!
#ifdef BC_MOTOR
        if(d_currentExperiment.motorScan().isEnabled())
        {
            d_waitingForMotor = true;
            emit motorRest();
            emit statusMessage(QString("Motor scan aborted. Returning motor to resting position..."));
            return;
        }
#endif
        finishAcquisition();
    }
}

void AcquisitionManager::auxDataTick()
{
    ps_currentExperiment->auxData()->startNewPoint();
    if(ps_currentExperiment->ftmwEnabled())
        processAuxData({{AuxDataStorage::makeKey("Ftmw","Shots"),
                         ps_currentExperiment->ftmwConfig()->completedShots()}});
    emit auxDataSignal();
}

#ifdef BC_MOTOR
void AcquisitionManager::motorMoveComplete(bool success)
{
    if(d_currentExperiment.isComplete() || d_currentExperiment.isAborted())
    {
        finishAcquisition();
        return;
    }

    //if motor move not successful, abort acquisition
    if(!success)
        abort();
    else
    {
        d_waitingForMotor = false;
    }
}

void AcquisitionManager::motorTraceReceived(const QVector<double> dat)
{
    if(d_state == Acquiring && !d_waitingForMotor && d_currentExperiment.motorScan().isEnabled())
    {
        bool adv = d_currentExperiment.addMotorTrace(dat);
        emit statusMessage(QString("Acquiring (%1/%2)").arg(d_currentExperiment.motorScan().currentPointShots())
                           .arg(d_currentExperiment.motorScan().shotsPerPoint()));
        if(adv)
        {
            checkComplete();

            if(d_state == Acquiring)
            {
                QVector3D pos = d_currentExperiment.motorScan().currentPos();
                emit startMotorMove(pos.x(),pos.y(),pos.z());
                emit statusMessage(QString("Moving motor to (X,Y,Z) = (%1, %2, %3)")
                                   .arg(pos.x(),0,'f',3).arg(pos.y(),0,'f',3).arg(pos.z(),0,'f',3));
                d_waitingForMotor = true;
            }

            emit motorDataUpdate(d_currentExperiment.motorScan());
        }

        //emit a progress signal
        emit motorProgress(d_currentExperiment.motorScan().completedShots());
    }

    //TODO: construct a rolling average waveform and send to UI
}
#endif

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
        {
#ifdef BC_MOTOR
            if(d_currentExperiment.motorScan().isEnabled())
            {
                d_state = Idle;
                d_waitingForMotor = true;
                emit motorRest();
                emit statusMessage(QString("Motor scan complete. Returning motor to resting position..."));
                return;
            }
#endif
            finishAcquisition();
        }
    }
}

void AcquisitionManager::finishAcquisition()
{
#ifdef BC_MOTOR
    d_waitingForMotor = false;
#endif
    emit endAcquisition();
    d_state = Idle;
    d_currentShift = 0;


    if(!ps_currentExperiment->isDummy())
    {
        emit statusMessage(QString("Saving experiment %1").arg(ps_currentExperiment->d_number));
        ps_currentExperiment->finalSave();
    }

    emit experimentComplete();
    ps_currentExperiment.reset();
}

bool AcquisitionManager::calculateShift(const QByteArray b)
{
    (void) b;
#pragma message("Implement calculateShift")
//    if(!d_currentExperiment.ftmwEnabled())
//        return true;

//    if(d_currentExperiment.ftmwConfig()->fidList().isEmpty())
//        return true;

//    if(d_currentExperiment.ftmwConfig()->completedShots() < 100)
//        return true;

//    //first, we need to extract the chirp from b
//    auto r = d_currentExperiment.ftmwConfig()->chirpRange();
//    QVector<qint64> newChirp = d_currentExperiment.ftmwConfig()->extractChirp(b);
//    if(newChirp.isEmpty())
//        return true;
//    Fid avgFid = d_currentExperiment.ftmwConfig()->fidList().constFirst();

//    int max = 5;
//    float thresh = 1.15; // fractional improvement needed to adjust shift
//    int shift = d_currentShift;
//    float fomCenter = calculateFom(newChirp,avgFid,r,shift);
//    float fomDown = calculateFom(newChirp,avgFid,r,shift-1);
//    float fomUp = calculateFom(newChirp,avgFid,r,shift+1);
//    bool done = false;
//    while(!done && qAbs(shift-d_currentShift) < max)
//    {
//        if(fomCenter > fomDown && fomCenter > fomUp)
//            done = true;
//        else if((fomDown-fomCenter) > (fomUp-fomCenter))
//        {
//            if(fomDown > thresh*fomCenter)
//            {
//                shift--;
//                fomUp = fomCenter;
//                fomCenter = fomDown;
//                fomDown = calculateFom(newChirp,avgFid,r,shift-1);
//            }
//            else
//                done = true;
//        }
//        else
//        {
//            if(fomUp > thresh*fomCenter)
//            {
//                shift++;
//                fomDown = fomCenter;
//                fomCenter = fomUp;
//                fomUp = calculateFom(newChirp,avgFid,r,shift+1);
//            }
//            else
//                done = true;
//        }
//    }

//    if(!done)
//    {
//        emit logMessage(QString("Calculated shift for this FID exceeded maximum permissible shift of %1 points. Fid rejected.").arg(max),LogHandler::Warning);
//        return false;
//    }

//    if(qAbs(d_currentShift - shift) > 0)
//    {
//        if(fomCenter < 0.9*d_lastFom)
//        {
//            emit logMessage(QString("Shot rejected. FOM (%1) is less than 90% of last FOM (%2)").arg(fomCenter,0,'e',2).arg(d_lastFom,0,'e',2));
//            return false;
//        }

//        emit logMessage(QString("Shift changed from %1 to %2. FOMs: (%3, %4, %5)").arg(d_currentShift).arg(shift)
//                        .arg(fomDown,0,'e',2).arg(fomCenter,0,'e',2).arg(fomUp,0,'e',2));
//        d_currentShift = shift;
////        return false;
//    }
//    if(qAbs(shift) > BC_FTMW_MAXSHIFT)
//    {
//        emit logMessage(QString("Total shift exceeds maximum range (%1). Aborting experiment.").arg(BC_FTMW_MAXSHIFT),LogHandler::Error);
//        abort();
//        return false;
//    }

//    d_lastFom = fomCenter;
//    return true;
    return true;

}

bool AcquisitionManager::scoreChirp(const QByteArray b)
{
    (void)b;
#pragma message("Implement scoreChirp")
//    if(!d_currentExperiment.ftmwEnabled())
//        return true;

//    if(d_currentExperiment.ftmwConfig()->fidList().isEmpty())
//        return true;

//    if(d_currentExperiment.ftmwConfig()->completedShots() < 20)
//        return true;

//    //Extract chirp from this waveform (1st frame)
//    QVector<qint64> newChirp = d_currentExperiment.ftmwConfig()->extractChirp(b);
//    if(newChirp.isEmpty())
//        return true;

//    //Calculate chirp RMS
//    double newChirpRMS = calculateChirpRMS(newChirp,d_currentExperiment.ftmwConfig()->fidTemplate().vMult());

//    //Get current RMS
//    QVector<qint64> currentChirp = d_currentExperiment.ftmwConfig()->extractChirp();
//    double currentRMS = calculateChirpRMS(currentChirp,d_currentExperiment.ftmwConfig()->fidTemplate().vMult(),d_currentExperiment.ftmwConfig()->completedShots());

////    emit logMessage(QString("This RMS: %1\tAVG RMS: %2").arg(newChirpRMS,0,'e',2).arg(currentRMS,0,'e',2));

//    //The chirp is good if its RMS is greater than threshold*currentRMS.
//    return newChirpRMS > currentRMS*d_currentExperiment.ftmwConfig()->chirpRMSThreshold();

    return true;

}

float AcquisitionManager::calculateFom(const QVector<qint64> vec, const Fid fid, QPair<int, int> range, int trialShift)
{
    //Kahan summation (32 bit precision is sufficient)
    float sum = 0.0;
    float c = 0.0;
    for(int i=0; i<vec.size(); i++)
    {
        if(i+range.first+trialShift >= 0 && i+range.first+trialShift < fid.size())
        {
            float dat = static_cast<float>(fid.atRaw(i+range.first+trialShift))*(static_cast<float>(vec.at(i)));
            float y = dat - c;
            float t = sum + y;
            c = (t-sum) - y;
            sum = t;
        }
    }

    return sum/static_cast<float>(fid.shots());
}

double AcquisitionManager::calculateChirpRMS(const QVector<qint64> chirp, double sf, qint64 shots)
{
    Q_UNUSED(sf)

    //Kahan summation
    double sum = 0.0;
    double c = 0.0;
    for(int i=0; i<chirp.size(); i++)
    {
        double dat = static_cast<double>(chirp.at(i)*chirp.at(i))/static_cast<double>(shots*shots);
        double y = dat - c;
        double t = sum + y;
        c = (t-sum) - y;
        sum = t;
    }

    return sqrt(sum);
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
