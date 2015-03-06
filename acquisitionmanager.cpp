#include "acquisitionmanager.h"
#include "oscilloscope.h"

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
    }

    //prepare data files, savemanager, fidmanager, etc
    d_currentExperiment = exp;
    d_state = Acquiring;
    emit logMessage(QString("Starting experiment %1.").arg(exp.number()),LogHandler::Highlight);
    emit statusMessage(QString("Acquiring"));
    emit beginAcquisition();

}

void AcquisitionManager::processScopeShot(const QByteArray b)
{
    static int total = 0;
    static int count = 0;
    if(d_state == Acquiring && d_currentExperiment.ftmwConfig().isEnabled())
    {
        d_testTime.restart();
        bool success = true;
        if(d_currentExperiment.ftmwConfig().fidList().isEmpty())
            success = d_currentExperiment.setFids(b);
        else
            success = d_currentExperiment.addFids(b);

        int t = d_testTime.elapsed();
        total += t;
        count++;
        emit logMessage(QString("Elapsed time: %1 ms, avg: %2").arg(t).arg(total/count));

        emit newFidList(d_currentExperiment.ftmwConfig().fidList());

        if(!success)
        {
            emit logMessage(d_currentExperiment.errorString(),LogHandler::Error);
            abort();
            return;
        }

        d_currentExperiment.incrementFtmw();
	   emit ftmwShotAcquired(d_currentExperiment.ftmwConfig().completedShots());
         //process shot, etc...
        checkComplete();
    }
}

void AcquisitionManager::pause()
{
    d_state = Paused;
    emit statusMessage(QString("Paused"));
}

void AcquisitionManager::resume()
{
    d_state = Acquiring;
    emit statusMessage(QString("Acquiring"));
}

void AcquisitionManager::abort()
{
    d_state = Idle;
    d_currentExperiment.setAborted();

    //save!
    finalSave();

    emit experimentComplete(d_currentExperiment);
}

void AcquisitionManager::checkComplete()
{
    if(d_state == Acquiring && d_currentExperiment.isComplete())
    {
        //do final save
        finalSave();
        d_state = Idle;
        emit experimentComplete(d_currentExperiment);
    }
}

void AcquisitionManager::finalSave()
{
    d_currentExperiment.save();
}

