#include "acquisitionmanager.h"

#include <savemanager.h>

AcquisitionManager::AcquisitionManager(QObject *parent) : QObject(parent), d_state(Idle), d_currentShift(0)
{
    p_saveThread = new QThread(this);
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
    if(!exp.hardwareSuccess() || exp.isDummy())
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

    d_currentShift = 0;
    //prepare data files, savemanager, fidmanager, etc
    d_currentExperiment = exp;

    SaveManager *sm = new SaveManager();
    connect(sm,&SaveManager::finalSaveComplete,p_saveThread,&QThread::quit);
    connect(sm,&SaveManager::finalSaveComplete,this,&AcquisitionManager::experimentComplete);
    connect(this,&AcquisitionManager::doFinalSave,sm,&SaveManager::finalSave);
    connect(this,&AcquisitionManager::takeSnapshot,sm,&SaveManager::snapshot);
    connect(p_saveThread,&QThread::finished,sm,&SaveManager::deleteLater);
    sm->moveToThread(p_saveThread);
    p_saveThread->start();

    d_state = Acquiring;
    emit logMessage(exp.startLogMessage(),BlackChirp::LogHighlight);
    emit statusMessage(QString("Acquiring"));

    if(d_currentExperiment.timeDataInterval() > 0)
    {
        if(d_timeDataTimer == nullptr)
            d_timeDataTimer = new QTimer(this);
        getTimeData();
        connect(d_timeDataTimer,&QTimer::timeout,this,&AcquisitionManager::getTimeData,Qt::UniqueConnection);
        d_timeDataTimer->start(d_currentExperiment.timeDataInterval()*1000);
    }
    d_currentExperiment.setAutoSaveShotsInterval(1000);
    emit beginAcquisition();

}

void AcquisitionManager::processFtmwScopeShot(const QByteArray b)
{
    static int total = 0;
    static int count = 0;
    if(d_state == Acquiring && d_currentExperiment.ftmwConfig().isEnabled()
            && !d_currentExperiment.ftmwConfig().isComplete())
    {

        QTime testTime;
        testTime.start();
        bool success = true;

        if(d_currentExperiment.ftmwConfig().isPhaseCorrectionEnabled() && d_currentExperiment.ftmwConfig().completedShots() > 50)
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

        int t = testTime.elapsed();
        total += t;
        count++;
        emit logMessage(QString("Elapsed time: %1 ms, avg: %2").arg(t).arg(total/count));

        d_currentExperiment.incrementFtmw();
        emit newFidList(d_currentExperiment.ftmwConfig().fidList());

        if(d_currentExperiment.ftmwConfig().type() == BlackChirp::FtmwTargetTime)
        {
            qint64 elapsedSecs = d_currentExperiment.startTime().secsTo(QDateTime::currentDateTime());
            emit ftmwShotAcquired(elapsedSecs);
        }
        else
            emit ftmwShotAcquired(d_currentExperiment.ftmwConfig().completedShots());

        checkComplete();
    }
}

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
            d_currentExperiment.addTimeData(l);
            emit timeData(l);
        }
    }
}

void AcquisitionManager::processTimeData(const QList<QPair<QString, QVariant> > timeDataList)
{
	if(d_state == Acquiring)
	{
		//test for abort conditions here!
		//
		//

//		emit logMessage(QString("Time data received."));
//		for(int i=0; i<timeDataList.size(); i++)
//			emit logMessage(QString("%1: %2").arg(timeDataList.at(i).first,timeDataList.at(i).second.toString()));

		d_currentExperiment.addTimeData(timeDataList);
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
        d_currentExperiment.setAborted();
        //save!
        finishAcquisition();
    }
}

void AcquisitionManager::checkComplete()
{
    if(d_state == Acquiring)
    {
        if(d_currentExperiment.snapshotReady())
            emit takeSnapshot(d_currentExperiment);

        if(d_currentExperiment.isComplete())
            finishAcquisition();
    }
}

void AcquisitionManager::finishAcquisition()
{
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

    if(d_currentExperiment.ftmwConfig().completedShots() < 50)
        return true;

    //first, we need to extract the chirp from b
    auto r = d_currentExperiment.ftmwConfig().chirpRange();
    if(r.first < 0 || r.second < 0)
        return true;

    QVector<qint16> newChirp(r.second-r.first);
    QVector<qint64> avgFid = d_currentExperiment.ftmwConfig().fidList().first().rawData();
    if(d_currentExperiment.ftmwConfig().scopeConfig().bytesPerPoint == 2)
    {
        for(int i=r.first; i<r.second; i++)
        {
            int index = i-r.first;
            qint8 b1 = b.at(2*i);
            qint8 b2 = b.at(2*i+1);
            qint16 dat = 0;
            if(d_currentExperiment.ftmwConfig().scopeConfig().byteOrder == QDataStream::LittleEndian)
            {
                dat |= b1;
                dat |= (b2 << 8);
            }
            else
            {
                dat |= (b1 << 8);
                dat |= b2;
            }
            newChirp[index] = dat;
        }
    }
    else
    {
        for(int i=r.first; i<r.second; i++)
        {
            int index = i-r.first;
            newChirp[index] = static_cast<qint16>(b.at(i));
        }
    }

    int max = 5;
    float thresh = 0.1; // fractional improvement needed to adjust shift
    int shift = d_currentShift;
    float fomCenter = calculateFom(newChirp,avgFid,r,shift);
    float fomDown = calculateFom(newChirp,avgFid,r,shift-1);
    float fomUp = calculateFom(newChirp,avgFid,r,shift+1);
    bool done = false;
    while(!done && qAbs(shift-d_currentShift) < max)
    {
        //always assume shift "wants" to go toward 0:
        if(shift >= 0)
        {
            if((fomDown-fomCenter) > qAbs(fomCenter)*thresh)
            {
                shift--;
                fomUp = fomCenter;
                fomCenter = fomDown;
                fomDown = calculateFom(newChirp,avgFid,r,shift-1);
            }
            else if((fomUp-fomCenter) > qAbs(fomCenter)*thresh)
            {
                shift++;
                fomDown = fomCenter;
                fomCenter = fomUp;
                fomUp = calculateFom(newChirp,avgFid,r,shift+1);
            }
            else
                done = true;
        }
        else
        {
            if((fomUp-fomCenter) > qAbs(fomCenter)*thresh)
            {
                shift++;
                fomDown = fomCenter;
                fomCenter = fomUp;
                fomUp = calculateFom(newChirp,avgFid,r,shift+1);
            }
            else if((fomDown-fomCenter) > qAbs(fomCenter)*thresh)
            {
                shift--;
                fomUp = fomCenter;
                fomCenter = fomDown;
                fomDown = calculateFom(newChirp,avgFid,r,shift-1);
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

    if(qAbs(shift) > BC_FTMW_MAXSHIFT)
    {
        emit logMessage(QString("Total shift exceeds maximum range (%1). Aborting experiment.").arg(BC_FTMW_MAXSHIFT),BlackChirp::LogError);
        abort();
        return false;
    }

    if(d_currentShift != shift)
    {
        emit logMessage(QString("Shift changed from %1 to %2. FOMs: (%3, %4, %5)").arg(d_currentShift).arg(shift)
                        .arg(fomDown,0,'e',2).arg(fomCenter,0,'e',2).arg(fomUp,0,'e',2));
    }
    d_currentShift = shift;
    return true;


}

float AcquisitionManager::calculateFom(const QVector<qint16> vec, const QVector<qint64> fid, QPair<int, int> range, int trialShift)
{
    //Kahan summation (32 bit precision is sufficient)
    float sum = 0.0;
    float c = 0.0;
    for(int i=0; i<vec.size(); i++)
    {
        if(i+range.first-trialShift >= 0 && i+range.first-trialShift < fid.size())
        {
            float dat = static_cast<float>(fid.at(i+range.first-trialShift)*vec.at(i));
            float y = dat - c;
            float t = sum + y;
            c = (t-sum) - y;
            sum = t;
        }
    }

    return sum;
}

