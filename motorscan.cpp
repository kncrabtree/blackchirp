#include "motorscan.h"

#include <QSettings>
#include <QApplication>

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

    s.setValue(QString("x0"),xVal(0));
    s.setValue(QString("y0"),yVal(0));
    s.setValue(QString("z0"),zVal(0));

    s.setValue(QString("dx"),(xVal(xPoints()-1) - xVal(0))/(static_cast<double>(xPoints())-1.0));
    s.setValue(QString("dy"),(yVal(yPoints()-1) - yVal(0))/(static_cast<double>(yPoints())-1.0));
    s.setValue(QString("dz"),(zVal(zPoints()-1) - zVal(0))/(static_cast<double>(zPoints())-1.0));

    s.setValue(QString("shotsPerPoint"),shotsPerPoint());

    s.endGroup();
    s.sync();
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

int MotorScan::numPoints(MotorScan::MotorDataAxis axis) const
{
    int out = 0;

    switch(axis) {
    case MotorX:
        out = xPoints();
        break;
    case MotorY:
        out = yPoints();
        break;
    case MotorZ:
        out = zPoints();
        break;
    case MotorT:
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

int MotorScan::shotsPerPoint() const
{
    return data->shotsPerPoint;
}

bool MotorScan::isComplete() const
{
    return data->currentPoint >= data->totalPoints;
}

QVector<QPointF> MotorScan::tSlice(int x, int y, int z) const
{
    QVector<QPointF> out(tPoints());
    for(int i=0; i<tPoints(); i++)
    {
        QPointF dat(tVal(i), data->zyxtData.at(z).at(y).at(x).at(i));
        out[i] = dat;
    }

    return out;

}

QVector<double> MotorScan::xySlice(int z, int t) const
{
    //row-x; column-y
    QVector<double> out(zPoints()*tPoints());
    int i =0;
    for(int y=0; y<yPoints(); y++)
    {
        for(int x=0; x<xPoints(); x++)
        {
            out[i] = data->zyxtData.at(z).at(y).at(x).at(t);
            i++;
        }
    }

    return out;
}

QVector<double> MotorScan::yzSlice(int x, int t) const
{
    //row-z; column-y
    QVector<double> out(zPoints()*yPoints());
    int i =0;
    for(int y=0; y<yPoints(); y++)
    {
        for(int z=0; z<zPoints(); z++)
        {
            out[i] = data->zyxtData.at(z).at(y).at(x).at(t);
            i++;
        }
    }

    return out;
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

void MotorScan::setShotsPerPoint(const int s)
{
    data->shotsPerPoint = s;
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

    data->currentPointShots = 0;
    data->currentPoint = 0;
    data->totalPoints = data->xPoints*data->yPoints*data->zPoints;
}


