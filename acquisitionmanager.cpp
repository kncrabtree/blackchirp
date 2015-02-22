#include "acquisitionmanager.h"

AcquisitionManager::AcquisitionManager(QObject *parent) : QObject(parent), d_state(Idle)
{

}

AcquisitionManager::~AcquisitionManager()
{

}

void AcquisitionManager::startExperiment(Experiment exp)
{
	if(!exp.isInitialized() || exp.isDummy())
		emit experimentComplete(exp);

    //prepare data files, savemanager, fidmanager, etc
    d_currentExperiment = exp;
    d_state = Acquiring;
    emit beginAcquisition();

}

void AcquisitionManager::processScopeShot(const QByteArray b)
{
    Q_UNUSED(b)

    if(d_state == Acquiring && d_currentExperiment.ftmwConfig().isEnabled())
    {
        d_currentExperiment.incrementFtmw();
	   emit ftmwShotAcquired(d_currentExperiment.ftmwConfig().completedShots());
         //process shot, etc...
        checkComplete();
    }
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

