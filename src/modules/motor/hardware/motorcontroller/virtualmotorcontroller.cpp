#include "virtualmotorcontroller.h"

#include <math.h>

#include "virtualinstrument.h"

VirtualMotorController::VirtualMotorController(QObject *parent) :
    MotorController(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual Motor Controller");
    d_commType = CommunicationProtocol::Virtual;
}

void VirtualMotorController::readSettings()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    double xHomeOffset = s.value(QString("xHomeOffset"),0).toDouble(); //total: about 77
    double yHomeOffset = s.value(QString("yHomeOffset"),80).toDouble(); //total: 162.516
    double zHomeOffset = s.value(QString("zHomeOffset"),200).toDouble(); //total: about 397
    s.setValue(QString("xHomeOffset"),xHomeOffset);
    s.setValue(QString("yHomeOffset"),yHomeOffset);
    s.setValue(QString("zHomeOffset"),zHomeOffset);

    double xMin = s.value(QString("xMin"),-100.0).toDouble();
    double xMax = s.value(QString("xMax"),100.0).toDouble();
    double yMin = s.value(QString("yMin"),-100.0).toDouble();
    double yMax = s.value(QString("yMax"),100.0).toDouble();
    double zMin = s.value(QString("zMin"),-100.0).toDouble();
    double zMax = s.value(QString("zMax"),100.0).toDouble();

    d_xRestingPos = s.value(QString("xRest"),xMin).toDouble();
    d_yRestingPos = s.value(QString("yRest"),yMin).toDouble();
    d_zRestingPos = s.value(QString("zRest"),zMin).toDouble();

    s.setValue(QString("xMin"),xMin);
    s.setValue(QString("xMax"),xMax);
    s.setValue(QString("yMin"),yMin);
    s.setValue(QString("yMax"),yMax);
    s.setValue(QString("zMin"),zMin);
    s.setValue(QString("zMax"),zMax);
    s.setValue(QString("xRest"),d_xRestingPos);
    s.setValue(QString("yRest"),d_yRestingPos);
    s.setValue(QString("zRest"),d_zRestingPos);

    d_xRange = qMakePair(xMin,xMax);
    d_yRange = qMakePair(yMin,yMax);
    d_zRange = qMakePair(zMin,zMax);

    s.endGroup();
    s.endGroup();
}



bool VirtualMotorController::testConnection()
{
    return true;
}

void VirtualMotorController::initialize()
{
}

bool VirtualMotorController::moveToPosition(double x, double y, double z)
{
    d_xPos = x;
    d_yPos = y;
    d_zPos = z;

    emit posUpdate(BlackChirp::MotorX,d_xPos);
    emit posUpdate(BlackChirp::MotorY,d_yPos);
    emit posUpdate(BlackChirp::MotorZ,d_zPos);

    emit motionComplete();
    return true;
}

void VirtualMotorController::moveToRestingPos()
{
    moveToPosition(d_xRestingPos,d_yRestingPos,d_zRestingPos);
}

void VirtualMotorController::checkLimit()
{
    bool nx = fabs(d_xPos-d_xRange.first) < 0.01;
    bool px = fabs(d_xPos-d_xRange.second) < 0.01;
    emit limitStatus(BlackChirp::MotorX,nx,px);
    bool ny = fabs(d_yPos-d_yRange.first) < 0.01;
    bool py = fabs(d_yPos-d_yRange.second) < 0.01;
    emit limitStatus(BlackChirp::MotorY,ny,py);
    bool nz = fabs(d_zPos-d_zRange.first) < 0.01;
    bool pz = fabs(d_zPos-d_zRange.second) < 0.01;
    emit limitStatus(BlackChirp::MotorZ,nz,pz);

}

bool VirtualMotorController::prepareForMotorScan(const MotorScan ms)
{
    Q_UNUSED(ms)
    return true;
}
