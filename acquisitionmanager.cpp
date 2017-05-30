#include "acquisitionmanager.h"

#include <savemanager.h>

AcquisitionManager::AcquisitionManager(QObject *parent) : QObject(parent), d_state(Idle), d_currentShift(0), d_lastFom(0.0)
{
    p_saveThread = new QThread(this);
#ifdef BC_MOTOR
    d_waitingForMotor = false;
#endif
}

AcquisitionManager::~AcquisitionManager()
{
    if(p_saveThread->isRunning())
    {
        p_saveThread->quit();
        p_saveThread->wait();
    }
}

void AcquisitionManager::beginExperiment(Experiment exp)
{
    if(!exp.hardwareSuccess())
    {
        if(!exp.errorString().isEmpty())
            emit logMessage(exp.errorString(),BlackChirp::LogError);
		emit experimentComplete(exp);
        return;
    }

#ifdef BC_CUDA
    if(exp.ftmwConfig().isEnabled())
    {
        //prepare GPU Averager
        BlackChirp::FtmwScopeConfig sc = exp.ftmwConfig().scopeConfig();
        bool success = gpuAvg.initialize(sc.recordLength,exp.ftmwConfig().numFrames(),
                                         sc.bytesPerPoint,sc.byteOrder);
        if(!success)
        {
            emit logMessage(gpuAvg.getErrorString(),BlackChirp::LogError);
            emit experimentComplete(exp);
            return;
        }
    }
#endif

    exp.setInitialized();
    if(!exp.isInitialized())
    {
        if(!exp.errorString().isEmpty())
            emit logMessage(exp.errorString(),BlackChirp::LogError);
        emit experimentComplete(exp);
        return;
    }

    emit experimentInitialized(exp);
    d_currentShift = 0;
    d_lastFom = 0.0;
    //prepare data files, savemanager, fidmanager, etc
    d_currentExperiment = exp;

    SaveManager *sm = new SaveManager();
    connect(sm,&SaveManager::finalSaveComplete,p_saveThread,&QThread::quit);
    connect(sm,&SaveManager::finalSaveComplete,this,&AcquisitionManager::experimentComplete);
    connect(this,&AcquisitionManager::doFinalSave,sm,&SaveManager::finalSave);
    connect(this,&AcquisitionManager::takeSnapshot,sm,&SaveManager::snapshot);
    connect(sm,&SaveManager::snapshotComplete,this,&AcquisitionManager::snapshotComplete);
    connect(p_saveThread,&QThread::finished,sm,&SaveManager::deleteLater);
    sm->moveToThread(p_saveThread);
    p_saveThread->start();

    d_state = Acquiring;
    emit statusMessage(QString("Acquiring"));

    if(d_currentExperiment.timeDataInterval() > 0)
    {
        if(d_timeDataTimer == nullptr)
            d_timeDataTimer = new QTimer(this);
        getTimeData();
        connect(d_timeDataTimer,&QTimer::timeout,this,&AcquisitionManager::getTimeData,Qt::UniqueConnection);
        d_timeDataTimer->start(d_currentExperiment.timeDataInterval()*1000);
    }
    emit beginAcquisition();

#ifdef BC_MOTOR
    if(d_currentExperiment.motorScan().isEnabled())
    {
        d_waitingForMotor = true;
        QVector3D pos = d_currentExperiment.motorScan().currentPos();
        emit startMotorMove(pos.x(),pos.y(),pos.z());
        emit statusMessage(QString("Moving motor to (X,Y,Z) = (%1, %2, %3)")
                           .arg(pos.x(),0,'f',3).arg(pos.y(),0,'f',3).arg(pos.z(),0,'f',3));
    }
#endif

}

void AcquisitionManager::processFtmwScopeShot(const QByteArray b)
{
//    static int total = 0;
//    static int count = 0;
    if(d_state == Acquiring && d_currentExperiment.ftmwConfig().isEnabled()
            && !d_currentExperiment.ftmwConfig().isComplete())
    {

//        QTime testTime;
//        testTime.start();
        bool success = true;

        if(d_currentExperiment.ftmwConfig().isPhaseCorrectionEnabled())
        {
            success = calculateShift(b);
            if(!success)
                return;
        }

#ifndef BC_CUDA
        success = d_currentExperiment.addFids(b,d_currentShift);
#else
        QList<QVector<qint64> >  l;
        if(d_currentExperiment.ftmwConfig().type() == BlackChirp::FtmwPeakUp)
            l = gpuAvg.parseAndRollAvg(b.constData(),d_currentExperiment.ftmwConfig().completedShots()+1,
                                       d_currentExperiment.ftmwConfig().targetShots(),d_currentShift);
        else
            l = gpuAvg.parseAndAdd(b.constData(),d_currentShift);

        if(l.isEmpty())
        {
            d_currentExperiment.setErrorString(gpuAvg.getErrorString());
            success = false;
        }
        else
            success = d_currentExperiment.setFidsData(l);
#endif

        if(!success)
        {
            emit logMessage(d_currentExperiment.errorString(),BlackChirp::LogError);
            abort();
            return;
        }

//        int t = testTime.elapsed();
//        total += t;
//        count++;
//        emit logMessage(QString("Elapsed time: %1 ms, avg: %2").arg(t).arg(total/count));

        d_currentExperiment.incrementFtmw();
        emit newFidList(d_currentExperiment.ftmwConfig().fidList());

        if(d_currentExperiment.ftmwConfig().type() == BlackChirp::FtmwTargetTime)
        {
            qint64 elapsedSecs = d_currentExperiment.startTime().secsTo(QDateTime::currentDateTime());
            emit ftmwUpdateProgress(elapsedSecs);
        }
        else
            emit ftmwUpdateProgress(d_currentExperiment.ftmwConfig().completedShots());

        emit ftmwNumShots(d_currentExperiment.ftmwConfig().completedShots());

        checkComplete();
    }
}

#ifdef BC_LIF
void AcquisitionManager::processLifScopeShot(const LifTrace t)
{
    if(d_state == Acquiring && d_currentExperiment.lifConfig().isEnabled() && !d_currentExperiment.isLifWaiting())
    {
        //process trace; only send data to UI if point is complete
        if(d_currentExperiment.addLifWaveform(t))
        {
            emit lifPointUpdate(d_currentExperiment.lifConfig().lastUpdatedLifPoint());

            if(!d_currentExperiment.isComplete())
            {
                d_currentExperiment.setLifWaiting(true);
                emit nextLifPoint(d_currentExperiment.lifConfig().currentDelay(),
                                  d_currentExperiment.lifConfig().currentFrequency());
            }
        }

        if(d_currentExperiment.lifConfig().completedShots() <= d_currentExperiment.lifConfig().totalShots())
            emit lifShotAcquired(d_currentExperiment.lifConfig().completedShots());

        checkComplete();
    }
}

void AcquisitionManager::lifHardwareReady(bool success)
{
    if(d_currentExperiment.lifConfig().isEnabled())
    {
        if(!success)
        {
            emit logMessage(QString("LIF delay and/or frequency could not be set. Aborting."),BlackChirp::LogError);
            abort();
        }
        else
            d_currentExperiment.setLifWaiting(false);
    }
}
#endif

void AcquisitionManager::changeRollingAverageShots(int newShots)
{
    if(d_state == Acquiring && d_currentExperiment.ftmwConfig().isEnabled() && d_currentExperiment.ftmwConfig().type() == BlackChirp::FtmwPeakUp)
        d_currentExperiment.overrideTargetShots(newShots);
}

void AcquisitionManager::resetRollingAverage()
{
    if(d_state == Acquiring && d_currentExperiment.ftmwConfig().isEnabled() && d_currentExperiment.ftmwConfig().type() == BlackChirp::FtmwPeakUp)
    {
        d_currentExperiment.resetFids();
#ifdef BC_CUDA
        gpuAvg.resetAverage();
#endif
    }
}

void AcquisitionManager::getTimeData()
{
    if(d_state == Acquiring)
    {
        emit timeDataSignal();

        d_currentExperiment.addTimeStamp();

        if(d_currentExperiment.ftmwConfig().isEnabled())
        {
            QList<QPair<QString,QVariant>> l { qMakePair(QString("ftmwShots"),d_currentExperiment.ftmwConfig().completedShots()) };
            d_currentExperiment.addTimeData(l,true);
            emit timeData(l,true);
        }
    }
}

void AcquisitionManager::processTimeData(const QList<QPair<QString, QVariant> > timeDataList, bool plot)
{
	if(d_state == Acquiring)
	{
        if(!d_currentExperiment.addTimeData(timeDataList,plot))
            abort();
    }
}

void AcquisitionManager::exportAsciiFid(const QString s)
{
    d_currentExperiment.exportAscii(s);
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
        d_currentExperiment.setAborted();
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
    if(d_state == Acquiring && !d_waitingForMotor)
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
        if(d_currentExperiment.snapshotReady())
            emit takeSnapshot(d_currentExperiment);

        if(d_currentExperiment.isComplete())
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

    disconnect(d_timeDataTimer,&QTimer::timeout,this,&AcquisitionManager::getTimeData);
    d_timeDataTimer->stop();

    emit doFinalSave(d_currentExperiment);
    emit statusMessage(QString("Saving experiment %1").arg(d_currentExperiment.number()));
}

bool AcquisitionManager::calculateShift(const QByteArray b)
{
    if(!d_currentExperiment.ftmwConfig().isEnabled())
        return true;

    if(d_currentExperiment.ftmwConfig().fidList().isEmpty())
        return true;

    if(d_currentExperiment.ftmwConfig().completedShots() < 100)
        return true;

    //first, we need to extract the chirp from b
    auto r = d_currentExperiment.ftmwConfig().chirpRange();
    QVector<qint64> newChirp = d_currentExperiment.ftmwConfig().extractChirp(b);
    if(newChirp.isEmpty())
        return true;
    Fid avgFid = d_currentExperiment.ftmwConfig().fidList().first();

    int max = 5;
    float thresh = 1.15; // fractional improvement needed to adjust shift
    int shift = d_currentShift;
    float fomCenter = calculateFom(newChirp,avgFid,r,shift);
    float fomDown = calculateFom(newChirp,avgFid,r,shift-1);
    float fomUp = calculateFom(newChirp,avgFid,r,shift+1);
    bool done = false;
    while(!done && qAbs(shift-d_currentShift) < max)
    {
        if(fomCenter > fomDown && fomCenter > fomUp)
            done = true;
        else if((fomDown-fomCenter) > (fomUp-fomCenter))
        {
            if(fomDown > thresh*fomCenter)
            {
                shift--;
                fomUp = fomCenter;
                fomCenter = fomDown;
                fomDown = calculateFom(newChirp,avgFid,r,shift-1);
            }
            else
                done = true;
        }
        else
        {
            if(fomUp > thresh*fomCenter)
            {
                shift++;
                fomDown = fomCenter;
                fomCenter = fomUp;
                fomUp = calculateFom(newChirp,avgFid,r,shift+1);
            }
            else
                done = true;
        }
    }

    if(!done)
    {
        emit logMessage(QString("Calculated shift for this FID exceeded maximum permissible shift of %1 points. Fid rejected.").arg(max),BlackChirp::LogWarning);
        return false;
    }

    if(qAbs(d_currentShift - shift) > 0)
    {
        if(fomCenter < 0.9*d_lastFom)
        {
            emit logMessage(QString("Shot rejected. FOM (%1) is less than 90% of last FOM (%2)").arg(fomCenter,0,'e',2).arg(d_lastFom,0,'e',2));
            return false;
        }

        emit logMessage(QString("Shift changed from %1 to %2. FOMs: (%3, %4, %5)").arg(d_currentShift).arg(shift)
                        .arg(fomDown,0,'e',2).arg(fomCenter,0,'e',2).arg(fomUp,0,'e',2));
        d_currentShift = shift;
//        return false;
    }
    if(qAbs(shift) > BC_FTMW_MAXSHIFT)
    {
        emit logMessage(QString("Total shift exceeds maximum range (%1). Aborting experiment.").arg(BC_FTMW_MAXSHIFT),BlackChirp::LogError);
        abort();
        return false;
    }

    d_lastFom = fomCenter;
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

