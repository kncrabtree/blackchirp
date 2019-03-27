#include "motorscan.h"

#include <QSettings>
#include <QApplication>
#include <QFile>
#include <math.h>

MotorScan::MotorScan() : data(new MotorScanData)
{

}

MotorScan::MotorScan(const MotorScan &rhs) : data(rhs.data)
{

}

MotorScan &MotorScan::operator=(const MotorScan &rhs)
{
    if (this != &rhs)
        data.operator=(rhs.data);
    return *this;
}

MotorScan MotorScan::fromSettings()
{
    MotorScan out;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("lastMotorScan"));

    out.setXPoints(s.value(QString("xPoints"),20).toInt());
    out.setYPoints(s.value(QString("yPoints"),20).toInt());
    out.setZPoints(s.value(QString("zPoints"),20).toInt());

    out.setShotsPerPoint(s.value(QString("shotsPerPoint"),10).toInt());

    double dx = s.value(QString("dx"),1.0).toDouble();
    double dy = s.value(QString("dy"),1.0).toDouble();
    double dz = s.value(QString("dz"),1.0).toDouble();
    double x0 = s.value(QString("x0"),-10.0).toDouble();
    double y0 = s.value(QString("y0"),-10.0).toDouble();
    double z0 = s.value(QString("z0"),-10.0).toDouble();

    out.setIntervals(x0,y0,z0,dx,dy,dz);

    BlackChirp::MotorScopeConfig ms;
    ms.dataChannel = s.value(QString("dataChannel"),1).toInt();
    ms.triggerChannel = s.value(QString("triggerChannel"),2).toInt();
    ms.recordLength = s.value(QString("recordLength"),500).toInt();
    ms.sampleRate = s.value(QString("sampleRate"),1e6).toDouble();
    ms.slope = static_cast<BlackChirp::ScopeTriggerSlope>(s.value(QString("slope"),BlackChirp::RisingEdge).toUInt());
    ms.verticalScale = s.value(QString("verticalScale"),1.0).toDouble();

    out.setScopeConfig(ms);

    s.endGroup();

    return out;
}


void MotorScan::saveToSettings() const
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("lastMotorScan"));

    s.setValue(QString("xPoints"),xPoints());
    s.setValue(QString("yPoints"),yPoints());
    s.setValue(QString("zPoints"),zPoints());
    s.setValue(QString("tPoints"),tPoints());

    s.setValue(QString("x0"),xVal(0));
    s.setValue(QString("y0"),yVal(0));
    s.setValue(QString("z0"),zVal(0));
    s.setValue(QString("t0"),tVal(0));

    s.setValue(QString("dx"),(xVal(xPoints()-1) - xVal(0))/(static_cast<double>(xPoints())-1.0));
    s.setValue(QString("dy"),(yVal(yPoints()-1) - yVal(0))/(static_cast<double>(yPoints())-1.0));
    s.setValue(QString("dz"),(zVal(zPoints()-1) - zVal(0))/(static_cast<double>(zPoints())-1.0));
    s.setValue(QString("dt"),(tVal(tPoints()-1) - tVal(0))/(static_cast<double>(tPoints())-1.0));

    s.setValue(QString("shotsPerPoint"),shotsPerPoint());

    s.setValue(QString("dataChannel"),scopeConfig().dataChannel);
    s.setValue(QString("triggerChannel"),scopeConfig().triggerChannel);
    s.setValue(QString("recordLength"),scopeConfig().recordLength);
    s.setValue(QString("sampleRate"),scopeConfig().sampleRate);
    s.setValue(QString("slope"),static_cast<uint>(scopeConfig().slope));
    s.setValue(QString("verticalScale"),scopeConfig().verticalScale);

    s.endGroup();

    s.sync();
}

bool MotorScan::isInitialized() const
{
    return data->initialized;
}

bool MotorScan::hardwareError() const
{
    return data->hardwareError;
}

bool MotorScan::isEnabled() const
{
    return data->enabled;
}

QMap<QString, QPair<QVariant, QString> > MotorScan::headerMap() const
{
    QMap<QString, QPair<QVariant, QString> > out;

    QString prefix = QString("MotorScan");
    QString empty = QString("");
    QString mm = QString("mm");
    QString us = QString::fromUtf16(u"μs");

    out.insert(prefix+QString("Enabled"),qMakePair(isEnabled(),empty));
    if(!isEnabled())
        return out;

    out.insert(prefix+QString("XPoints"),qMakePair(xPoints(),empty));
    out.insert(prefix+QString("YPoints"),qMakePair(yPoints(),empty));
    out.insert(prefix+QString("ZPoints"),qMakePair(zPoints(),empty));
    out.insert(prefix+QString("TPoints"),qMakePair(tPoints(),empty));

    out.insert(prefix+QString("XStart"),qMakePair(data->x0,mm));
    out.insert(prefix+QString("YStart"),qMakePair(data->y0,mm));
    out.insert(prefix+QString("ZStart"),qMakePair(data->z0,mm));
    out.insert(prefix+QString("TStart"),qMakePair(data->t0,us));

    out.insert(prefix+QString("XStep"),qMakePair(data->dx,mm));
    out.insert(prefix+QString("YStep"),qMakePair(data->dy,mm));
    out.insert(prefix+QString("ZStep"),qMakePair(data->dz,mm));
    out.insert(prefix+QString("TStep"),qMakePair(data->dt,us));

    out.insert(prefix+QString("ShotsPerPoint"),qMakePair(data->shotsPerPoint,empty));

    out.unite(data->scopeConfig.headerMap());
    return out;

}

int MotorScan::xPoints() const
{
    return data->xPoints;
}

int MotorScan::yPoints() const
{
    return data->yPoints;
}

int MotorScan::zPoints() const
{
    return data->zPoints;
}

int MotorScan::tPoints() const
{
    return data->tPoints;
}

int MotorScan::numPoints(BlackChirp::MotorAxis axis) const
{
    int out = 0;

    switch(axis) {
    case BlackChirp::MotorX:
        out = xPoints();
        break;
    case BlackChirp::MotorY:
        out = yPoints();
        break;
    case BlackChirp::MotorZ:
        out = zPoints();
        break;
    case BlackChirp::MotorT:
        out = tPoints();
        break;
    }

    return out;
}

double MotorScan::xVal(int i) const
{
    if(i<0 || i>=xPoints())
        return -1.0;

    return data->x0 + static_cast<double>(i)*data->dx;
}

double MotorScan::yVal(int i) const
{
    if(i<0 || i>=yPoints())
        return -1.0;

    return data->y0 + static_cast<double>(i)*data->dy;
}

double MotorScan::zVal(int i) const
{
    if(i<0 || i>=zPoints())
        return -1.0;

    return data->z0 + static_cast<double>(i)*data->dz;
}

double MotorScan::tVal(int i) const
{
    if(i<0 || i>=tPoints())
        return -1.0;

    return data->t0 + static_cast<double>(i)*data->dt;
}

double MotorScan::axisValue(BlackChirp::MotorAxis axis, int i) const
{
    if(i<0 || i >= numPoints(axis))
        return -1.0;

    double out = -1.0;

    switch(axis) {
    case BlackChirp::MotorX:
        out = xVal(i);
        break;
    case BlackChirp::MotorY:
        out = yVal(i);
        break;
    case BlackChirp::MotorZ:
        out = zVal(i);
        break;
    case BlackChirp::MotorT:
        out = tVal(i);
        break;
    }

    return out;
}

QPair<double, double> MotorScan::range(BlackChirp::MotorAxis axis) const
{
    double first = axisValue(axis,0);
    double last = axisValue(axis,numPoints(axis)-1);
    return qMakePair(first,last);
}

QPair<double, double> MotorScan::interval(BlackChirp::MotorAxis axis) const
{
    double first = axisValue(axis,0);
    double last = axisValue(axis,numPoints(axis)-1);
    double halfStep = fabs(first - last)/static_cast<double>(numPoints(axis)-1)/2.0;
    double min = qMin(first,last) - halfStep;
    double max = qMax(first,last) + halfStep;

    return qMakePair(min,max);

}

double MotorScan::value(int x, int y, int z, int t) const
{
    return data->zyxtData.at(z).at(y).at(x).at(t);
}

double MotorScan::smoothValue(int x, int y, int z, int t, Eigen::MatrixXd coefs) const
{
    return Analysis::savGolSmoothPoint(t,coefs,0,data->zyxtData.at(z).at(y).at(x));
}

bool MotorScan::writeMotorFile(int num) const
{
    QFile motor(BlackChirp::getExptFile(num,BlackChirp::MotorFile));
    if(motor.open(QIODevice::WriteOnly))
    {
        QDataStream d(&motor);
        d << QByteArray("BCMOTORv1.0");
        d << data->zyxtData;
        motor.close();
        return true;
    }
    else
        return false;
}

int MotorScan::shotsPerPoint() const
{
    return data->shotsPerPoint;
}

int MotorScan::currentPointShots() const
{
    return data->currentPointShots;
}

int MotorScan::completedShots() const
{
    return data->currentPoint*data->shotsPerPoint + data->currentPointShots;
}

bool MotorScan::isPointComplete() const
{
    return data->currentPointShots >= data->shotsPerPoint;
}

bool MotorScan::isComplete() const
{
    return data->currentPoint >= data->totalPoints;
}

QVector3D MotorScan::currentPos() const
{
    return QVector3D(xVal(data->currentX),yVal(data->currentY),zVal(data->currentZ));
}

QVector<double> MotorScan::slice(BlackChirp::MotorAxis xAxis, BlackChirp::MotorAxis yAxis, BlackChirp::MotorAxis otherAxis1, int otherPoint1, BlackChirp::MotorAxis otherAxis2, int otherPoint2) const
{

    if(xAxis == yAxis || xAxis == otherAxis1 || xAxis == otherAxis2 || yAxis == otherAxis1 ||
            yAxis == otherAxis2 || otherAxis1 == otherAxis2)
        return QVector<double>();

    int i,j, k=otherPoint1, l=otherPoint2;
    int *x, *y, *z, *t;

    switch(xAxis)
    {
    case BlackChirp::MotorX:
        x = &i;
        break;
    case BlackChirp::MotorY:
        y = &i;
        break;
    case BlackChirp::MotorZ:
        z = &i;
        break;
    case BlackChirp::MotorT:
        t = &i;
        break;
    }

    switch(yAxis)
    {
    case BlackChirp::MotorX:
        x = &j;
        break;
    case BlackChirp::MotorY:
        y = &j;
        break;
    case BlackChirp::MotorZ:
        z = &j;
        break;
    case BlackChirp::MotorT:
        t = &j;
        break;
    }

    switch(otherAxis1)
    {
    case BlackChirp::MotorX:
        x = &k;
        break;
    case BlackChirp::MotorY:
        y = &k;
        break;
    case BlackChirp::MotorZ:
        z = &k;
        break;
    case BlackChirp::MotorT:
        t = &k;
        break;
    }

    switch(otherAxis2)
    {
    case BlackChirp::MotorX:
        x = &l;
        break;
    case BlackChirp::MotorY:
        y = &l;
        break;
    case BlackChirp::MotorZ:
        z = &l;
        break;
    case BlackChirp::MotorT:
        t = &l;
        break;
    }

    QVector<double> out(numPoints(xAxis)*numPoints(yAxis));
    int idx = 0;
    for(i=0; i<numPoints(xAxis); i++)
    {
        for(j=0; j<numPoints(yAxis); j++)
        {
            out[idx] = value(*x,*y,*z,*t);
            idx++;
        }
    }

    return out;

}

QVector<double> MotorScan::smoothSlice(BlackChirp::MotorAxis xAxis, BlackChirp::MotorAxis yAxis, BlackChirp::MotorAxis otherAxis1, int otherPoint1, BlackChirp::MotorAxis otherAxis2, int otherPoint2, Eigen::MatrixXd coefs) const
{

    if(xAxis == yAxis || xAxis == otherAxis1 || xAxis == otherAxis2 || yAxis == otherAxis1 ||
            yAxis == otherAxis2 || otherAxis1 == otherAxis2)
        return QVector<double>();

    int i,j, k=otherPoint1, l=otherPoint2;
    int *x, *y, *z, *t;

    switch(xAxis)
    {
    case BlackChirp::MotorX:
        x = &i;
        break;
    case BlackChirp::MotorY:
        y = &i;
        break;
    case BlackChirp::MotorZ:
        z = &i;
        break;
    case BlackChirp::MotorT:
        t = &i;
        break;
    }

    switch(yAxis)
    {
    case BlackChirp::MotorX:
        x = &j;
        break;
    case BlackChirp::MotorY:
        y = &j;
        break;
    case BlackChirp::MotorZ:
        z = &j;
        break;
    case BlackChirp::MotorT:
        t = &j;
        break;
    }

    switch(otherAxis1)
    {
    case BlackChirp::MotorX:
        x = &k;
        break;
    case BlackChirp::MotorY:
        y = &k;
        break;
    case BlackChirp::MotorZ:
        z = &k;
        break;
    case BlackChirp::MotorT:
        t = &k;
        break;
    }

    switch(otherAxis2)
    {
    case BlackChirp::MotorX:
        x = &l;
        break;
    case BlackChirp::MotorY:
        y = &l;
        break;
    case BlackChirp::MotorZ:
        z = &l;
        break;
    case BlackChirp::MotorT:
        t = &l;
        break;
    }

    QVector<double> out(numPoints(xAxis)*numPoints(yAxis));
    int idx = 0;
    for(i=0; i<numPoints(xAxis); i++)
    {
        for(j=0; j<numPoints(yAxis); j++)
        {
            out[idx] = smoothValue(*x,*y,*z,*t,coefs);
            idx++;
        }
    }

    return out;
}

QVector<QPointF> MotorScan::tTrace(int x, int y, int z) const
{
    QVector<QPointF> out(tPoints());
    for(int t=0; t<tPoints(); t++)
    {
        QPointF dat(tVal(t), value(x,y,z,t));
        out[t] = dat;
    }

    return out;

}

QVector<QPointF> MotorScan::smoothtTrace(int x, int y, int z, Eigen::MatrixXd coefs) const
{
    QVector<QPointF> out(tPoints());
    QVector<double> tSmooth = Analysis::savGolSmooth(coefs,0,data->zyxtData.at(z).at(y).at(x));
    for(int t=0; t<tPoints(); t++)
        out[t] = QPointF(tVal(t),tSmooth.at(t));

    return out;
}

BlackChirp::MotorScopeConfig MotorScan::scopeConfig() const
{
    return data->scopeConfig;
}

void MotorScan::parseLine(QString key, QVariant val)
{
    if(key.startsWith(QString("MotorScan")))
    {
        if(key.endsWith(QString("Enabled")))
            data->enabled = val.toBool();
        if(key.endsWith(QString("XPoints")))
            data->xPoints = val.toInt();
        if(key.endsWith(QString("YPoints")))
            data->yPoints = val.toInt();
        if(key.endsWith(QString("ZPoints")))
            data->zPoints = val.toInt();
        if(key.endsWith(QString("TPoints")))
            data->tPoints = val.toInt();
        if(key.endsWith(QString("XStart")))
            data->x0 = val.toDouble();
        if(key.endsWith(QString("YStart")))
            data->y0 = val.toDouble();
        if(key.endsWith(QString("ZStart")))
            data->z0 = val.toDouble();
        if(key.endsWith(QString("TStart")))
            data->t0 = val.toDouble();
        if(key.endsWith(QString("XStep")))
            data->dx = val.toDouble();
        if(key.endsWith(QString("YStep")))
            data->dy = val.toDouble();
        if(key.endsWith(QString("ZStep")))
            data->dz = val.toDouble();
        if(key.endsWith(QString("TStep")))
            data->dt = val.toDouble();
        if(key.endsWith(QString("ShotsPerPoint")))
            data->shotsPerPoint = val.toInt();
    }
    else if(key.startsWith(QString("MotorScope")))
    {
        if(key.endsWith(QString("VerticalScale")))
            data->scopeConfig.verticalScale = val.toDouble();
        if(key.endsWith(QString("TriggerSlope")))
        {
            if(val.toString().contains(QString("Rising")))
                data->scopeConfig.slope = BlackChirp::RisingEdge;
            else
                data->scopeConfig.slope = BlackChirp::FallingEdge;
        }
        if(key.endsWith(QString("SampleRate")))
            data->scopeConfig.sampleRate = val.toDouble()*1e6;
        if(key.endsWith(QString("RecordLength")))
            data->scopeConfig.recordLength = val.toInt();
        if(key.endsWith(QString("BytesPerPoint")))
            data->scopeConfig.bytesPerPoint = val.toInt();
        if(key.endsWith(QString("ByteOrder")))
        {
            if(val.toString().contains(QString("Big")))
                data->scopeConfig.byteOrder = QDataStream::BigEndian;
            else
                data->scopeConfig.byteOrder = QDataStream::LittleEndian;
        }
        if(key.endsWith(QString("TriggerChannel")))
            data->scopeConfig.triggerChannel = val.toInt();
        if(key.endsWith(QString("DataChannel")))
            data->scopeConfig.dataChannel = val.toInt();
    }
}

void MotorScan::setEnabled(bool en)
{
    data->enabled = en;
}

void MotorScan::setXPoints(int x)
{
    data->xPoints = x;
}

void MotorScan::setYPoints(int y)
{
    data->yPoints = y;
}

void MotorScan::setZPoints(int z)
{
    data->zPoints = z;
}

void MotorScan::setTPoints(int t)
{
    data->tPoints = t;
}

void MotorScan::setIntervals(double x0, double y0, double z0, double dx, double dy, double dz)
{
    data->x0 = x0;
    data->y0 = y0;
    data->z0 = z0;

    data->dx = dx;
    data->dy = dy;
    data->dz = dz;
}

bool MotorScan::loadMotorData(int num, const QString path)
{
    QFile motor(BlackChirp::getExptFile(num,BlackChirp::MotorFile,path));
    if(motor.open(QIODevice::ReadOnly))
    {
        QDataStream d(&motor);
        QByteArray magic;
        d >> magic;
        if(!magic.startsWith("BCMOTOR"))
        {
            motor.close();
            return false;
        }
        if(magic.endsWith("v1.0"))
        {
            QList<QList<QList<QVector<double>>>> l;
            d >> l;
            data->zyxtData = l;
        }

        motor.close();
        return true;
    }

    return false;
}

void MotorScan::setShotsPerPoint(const int s)
{
    data->shotsPerPoint = s;
}

void MotorScan::setScopeConfig(const BlackChirp::MotorScopeConfig &sc)
{
    data->scopeConfig = sc;
    data->tPoints = sc.recordLength;
    data->dt = 1.0/sc.sampleRate*1e6;
}

void MotorScan::setHardwareError()
{
    data->hardwareError = true;
}

void MotorScan::initialize()
{
    for(int z=0; z<zPoints(); z++)
    {
        QList<QList<QVector<double>>> yd;
        for(int y=0; y<yPoints(); y++)
        {
            QList<QVector<double>> xd;
            for(int x=0; x<xPoints(); x++)
            {
                QVector<double> tDat(tPoints());
                xd.append(tDat);
            }
            yd.append(xd);
        }
        data->zyxtData.append(yd);
    }

    data->initialized = true;
    data->currentPointShots = 0;
    data->currentPoint = 0;
    data->totalPoints = data->xPoints*data->yPoints*data->zPoints;
    data->currentX = 0;
    data->currentY = 0;
    data->currentZ = 0;
}

bool MotorScan::addTrace(const QVector<double> d)
{
    QVector<double> newDat = data->zyxtData.at(data->currentZ).at(data->currentY).at(data->currentX);

    data->currentPointShots++;
    bool adv = isPointComplete();
    if(!adv)
    {
        for(int i=0; i<newDat.size() && i < d.size(); i++)
            newDat[i] += d.at(i);
    }
    else
    {
        double shots = static_cast<double>(data->shotsPerPoint);
        for(int i=0; i<newDat.size() && i < d.size(); i++)
            newDat[i] = (newDat.at(i) + d.at(i))/shots;
    }

    data->zyxtData[data->currentZ][data->currentY][data->currentX] = newDat;
    if(adv)
        advance();

    return adv;
}

void MotorScan::advance()
{
    data->currentPoint++;

    data->currentX++;
    if(data->currentX == xPoints())
    {
        data->currentX = 0;
        data->currentY++;
        if(data->currentY == yPoints())
        {
            data->currentY = 0;
            data->currentZ++;
        }
    }

    data->currentPointShots = 0;
}

void MotorScan::abort()
{
    if(data->currentPointShots < data->shotsPerPoint && data->currentPointShots > 0)
    {
        for(int i=0; i<tPoints(); i++)
            data->zyxtData[data->currentZ][data->currentY][data->currentX][i]/=static_cast<double>(data->currentPointShots);
    }
}


