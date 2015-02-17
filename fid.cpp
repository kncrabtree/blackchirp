#include "fid.h"
#include <QSharedData>

//this is all pretty straightforward

/*!
 \brief Internal data for Fid

 Stores the time spacing between points (in s), the probe frequency (in MHz), and the FID data (arbitrary units)
*/
class FidData : public QSharedData {
public:
/*!
 \brief Default constructor

*/
    FidData() : spacing(5e-7), probeFreq(0.0), fid(QVector<double>(400)), sideband(Fid::UpperSideband) {}
/*!
 \brief Copy constructor

 \param other Object to copy
*/
    FidData(const FidData &other) : QSharedData(other), spacing(other.spacing), probeFreq(other.probeFreq), fid(other.fid),
        sideband(other.sideband) {}
    ~FidData(){}

    double spacing;
    double probeFreq;
    QVector<double> fid;
    Fid::Sideband sideband;
};

Fid::Fid() : data(new FidData)
{
}

Fid::Fid(const Fid &rhs) : data(rhs.data)
{
}

Fid::Fid(const double sp, const double p, const QVector<double> d, Sideband sb)
{
    data = new FidData;
    data->spacing = sp;
    data->probeFreq = p;
    data->fid = d;
    data->sideband = sb;
}

Fid &Fid::operator=(const Fid &rhs)
{
    if (this != &rhs)
        data.operator=(rhs.data);
    return *this;
}

Fid::~Fid()
{
}

void Fid::setSpacing(const double s)
{
    data->spacing = s;
}

void Fid::setProbeFreq(const double f)
{
    data->probeFreq = f;
}

void Fid::setData(const QVector<double> d)
{
    data->fid = d;
}

void Fid::setSideband(const Fid::Sideband sb)
{
    data->sideband = sb;
}

int Fid::size() const
{
    return data->fid.size();
}

double Fid::at(const int i) const
{
    return data->fid.at(i);
}

double Fid::spacing() const
{
    return data->spacing;
}

double Fid::probeFreq() const
{
    return data->probeFreq;
}

QVector<QPointF> Fid::toXY() const
{
    //create a vector of points that can be displayed
    QVector<QPointF> out;
    out.reserve(size());

    for(int i=0; i<size(); i++)
        out.append(QPointF(spacing()*(double)i,at(i)));

    return out;
}

QVector<double> Fid::toVector() const
{
    return data->fid;
}

Fid::Sideband Fid::sideband() const
{
    return data->sideband;
}

double Fid::maxFreq() const
{
    if(spacing()>0.0)
    {
        if(sideband() == UpperSideband)
            return probeFreq() + 1.0/2.0/spacing()/1e6;
        else
            return probeFreq();
    }
    else
        return 0.0;
}

double Fid::minFreq() const
{
    if(spacing()>0.0)
    {
        if(sideband() == LowerSideband)
            return probeFreq() - 1.0/2.0/spacing()/1e6;
        else
            return probeFreq();
    }
    else
        return 0.0;
}
