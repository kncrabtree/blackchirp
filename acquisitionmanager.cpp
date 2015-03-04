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
		emit experimentComplete(exp);

    //prepare data files, savemanager, fidmanager, etc
    d_currentExperiment = exp;
    d_state = Acquiring;
    emit logMessage(QString("Starting experiment %1.").arg(exp.number()),LogHandler::Highlight);
    emit statusMessage(QString("Acquiring"));
    emit beginAcquisition();

}

void AcquisitionManager::processScopeShot(const QByteArray b)
{
    if(d_state == Acquiring && d_currentExperiment.ftmwConfig().isEnabled())
    {
        d_testTime.restart();
        QVector<qint64> newData = Oscilloscope::parseWaveform(b,d_currentExperiment.ftmwConfig().scopeConfig());

        if(d_currentExperiment.ftmwConfig().fidList().isEmpty())
            d_currentExperiment.setFids(newData);
        else
            d_currentExperiment.addFids(newData);

        int t = d_testTime.elapsed();
        emit logMessage(QString("Elapsed time: %1 ms").arg(t));

        emit newFidList(d_currentExperiment.ftmwConfig().fidList());

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

    emit experimentComplete(d_currentExperiment);
}

void AcquisitionManager::checkComplete()
{
    if(d_state == Acquiring && d_currentExperiment.isComplete())
    {
        //do final save
        d_state = Idle;
        emit experimentComplete(d_currentExperiment);
    }
}

