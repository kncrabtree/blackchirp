#include <data/experiment/fid.h>

#include <data/analysis/analysis.h>
#include <data/experiment/digitizerconfig.h>

Fid::Fid() : data(new FidData)
{
}

Fid::Fid(const Fid &rhs) : data(rhs.data)
{
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

void Fid::setData(const QVector<qint64> d)
{
    data->fid = d;
}

void Fid::setSideband(const RfConfig::Sideband sb)
{
    data->sideband = sb;
}

void Fid::setVMult(const double vm)
{
    data->vMult = vm;
}

void Fid::setShots(const quint64 s)
{
    data->shots = s;
}

void Fid::detach()
{
    data->fid.detach();
}

Fid &Fid::operator +=(const Fid other)
{
    Q_ASSERT(size() == other.size());
    for(int i=0;i<size();i++)
        data->fid[i] += other.atRaw(i);

    data->shots += other.shots();

    return *this;
}

Fid &Fid::operator +=(const QVector<qint64> other)
{
    Q_ASSERT(size() == other.size());
    for(int i=0; i<size();i++)
        data->fid[i] += other.at(i);

    data->shots++;

    return *this;
}

Fid &Fid::operator +=(const qint64 *other)
{
    for(int i=0; i<size();i++)
        data->fid[i] += other[i];

    data->shots++;

    return *this;
}

Fid &Fid::operator -=(const Fid other)
{
    Q_ASSERT(data->shots > other.shots());
    Q_ASSERT(size() == other.size());

    for(int i=0; i<size(); i++)
        data->fid[i] -= other.atRaw(i);

    data->shots -= other.shots();

    return *this;
}

void Fid::add(const qint64 *other, const unsigned int offset)
{
    for(int i=0; i<size(); i++)
        data->fid[i] += other[i+offset];

    data->shots++;
}

void Fid::add(const Fid other, int shift)
{
    Q_ASSERT(size() == other.size());
    if(shift == 0)
        *this += other;
    else
    {
        for(int i=0; i<size(); i++)
        {
            if(i+shift >=0 && i >= 0 && i+shift < size() && i < size())
                data->fid[i+shift] += other.atRaw(i);
        }

        data->shots += other.shots();
    }
}

void Fid::copyAdd(const qint64 *other, const unsigned int offset)
{
    memcpy(data->fid.data(),other + offset,sizeof(qint64)*size());
    data->shots++;
}

void Fid::rollingAverage(const Fid other, quint64 targetShots, int shift)
{
    Q_ASSERT(size() == other.size());
    quint64 totalShots = shots() + other.shots();
    if(totalShots <= targetShots)
        add(other,shift);
    else
    {
        for(int i=0; i<other.size(); i++)
        {
            if(i+shift >=0 && i+shift < size())
                data->fid[i+shift] = Analysis::intRoundClosest(targetShots*(atRaw(i+shift) + other.atRaw(i)),totalShots);
            else
                data->fid[i] = Analysis::intRoundClosest(targetShots*atRaw(i),totalShots);
        }

        data->shots = targetShots;
    }
}

int Fid::size() const
{
    return data->fid.size();
}

bool Fid::isEmpty() const
{
    return data->fid.isEmpty();
}

double Fid::at(const int i) const
{
    return atNorm(i)*data->vMult;
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
    QVector<double> out;
    out.reserve(size());
    for(int i=0;i<size();i++)
        out.append(at(i));

    return out;
}

QVector<qint64> Fid::rawData() const
{
    return data->fid;
}

qint64 Fid::atRaw(const int i) const
{
    return data->fid.at(i);
}

double Fid::atNorm(const int i) const
{
    if(data->shots > 1)
        return ((double)data->fid.at(i)/(double)data->shots);
    else
        return (double)data->fid.at(i);
}

quint64 Fid::shots() const
{
    return data->shots;
}

RfConfig::Sideband Fid::sideband() const
{
    return data->sideband;
}

double Fid::maxFreq() const
{
    if(spacing()>0.0)
    {
        if(sideband() == RfConfig::UpperSideband)
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
        if(sideband() == RfConfig::LowerSideband)
            return probeFreq() - 1.0/2.0/spacing()/1e6;
        else
            return probeFreq();
    }
    else
        return 0.0;
}

double Fid::vMult() const
{
    return data->vMult;
}
