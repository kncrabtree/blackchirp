#include "acquisitionmanager.h"
#include "ftmwscope.h"

AcquisitionManager::AcquisitionManager(QObject *parent) : QObject(parent), d_state(Idle)
{

}

AcquisitionManager::~AcquisitionManager()
{

}

void AcquisitionManager::beginExperiment(Experiment exp)
{
	if(!exp.isInitialized() || exp.isDummy())
    {
        if(!exp.errorString().isEmpty())
            emit logMessage(exp.errorString(),LogHandler::Error);
		emit experimentComplete(exp);
        return;
    }

    //prepare data files, savemanager, fidmanager, etc
    d_currentExperiment = exp;

    d_state = Acquiring;
    emit logMessage(QString("Starting experiment %1.").arg(exp.number()),LogHandler::Highlight);
    emit statusMessage(QString("Acquiring"));

    if(d_currentExperiment.timeDataInterval() > 0)
    {
        getTimeData();
        connect(&d_timeDataTimer,&QTimer::timeout,this,static_cast<void (AcquisitionManager::*)(void)>(&AcquisitionManager::getTimeData),Qt::UniqueConnection);
        d_timeDataTimer.start(d_currentExperiment.timeDataInterval()*1000);
    }
    emit beginAcquisition();

}

void AcquisitionManager::processScopeShot(const QByteArray b)
{
//    static int total = 0;
//    static int count = 0;
    if(d_state == Acquiring && d_currentExperiment.ftmwConfig().isEnabled())
    {
        d_testTime.restart();
        bool success = true;
        if(d_currentExperiment.ftmwConfig().fidList().isEmpty())
            success = d_currentExperiment.setFids(b);
        else
            success = d_currentExperiment.addFids(b);

        if(!success)
        {
            emit logMessage(d_currentExperiment.errorString(),LogHandler::Error);
            abort();
            return;
        }

//        int t = d_testTime.elapsed();
//        total += t;
//        count++;
//        emit logMessage(QString("Elapsed time: %1 ms, avg: %2").arg(t).arg(total/count));

        d_currentExperiment.incrementFtmw();
        emit newFidList(d_currentExperiment.ftmwConfig().fidList());
        emit ftmwShotAcquired(d_currentExperiment.ftmwConfig().completedShots());

        checkComplete();
    }
}

void AcquisitionManager::getTimeData()
{
    if(d_state == Acquiring)
    {
        emit timeDataSignal();

        d_currentExperiment.addTimeStamp();
        emit logMessage(QString("Timestamp: %1").arg(QDateTime::currentDateTime().toString()));

        if(d_currentExperiment.ftmwConfig().isEnabled())
        {
            d_currentExperiment.addTimeData(QList<QPair<QString,QVariant>>{ qMakePair(QString("ftmwShots"),d_currentExperiment.ftmwConfig().completedShots()) } );
            emit logMessage(QString("ftmwShots: %1").arg(d_currentExperiment.ftmwConfig().completedShots()));
        }
    }
}

void AcquisitionManager::processTimeData(const QList<QPair<QString, QVariant> > timeDataList)
{
    //test for abort conditions here!
    //
    //

    emit logMessage(QString("Time data received."));
    for(int i=0; i<timeDataList.size(); i++)
        emit logMessage(QString("%1: %2").arg(timeDataList.at(i).first,timeDataList.at(i).second.toString()));

    d_currentExperiment.addTimeData(timeDataList);
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
        endAcquisition();
    }
}

void AcquisitionManager::checkComplete()
{
    if(d_state == Acquiring && d_currentExperiment.isComplete())
    {
        //do final save
        endAcquisition();
    }
}

void AcquisitionManager::endAcquisition()
{
    d_state = Idle;

    disconnect(&d_timeDataTimer,&QTimer::timeout,this,static_cast<void (AcquisitionManager::*)(void)>(&AcquisitionManager::getTimeData));
    d_timeDataTimer.stop();
    d_currentExperiment.save();

    emit experimentComplete(d_currentExperiment);
}

