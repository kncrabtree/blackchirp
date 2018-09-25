#include "ft.h"

class FtData : public QSharedData
{
public:
    FtData() : yMin(0.0), yMax(0.0), loFreq(0.0) {}

    double yMin;
    double yMax;
    double loFreq;
    QVector<QPointF> ftData;
};

Ft::Ft() : data(new FtData)
{

}

Ft::Ft(int numPnts, double loFreq) : data(new FtData)
{
    data->ftData.resize(numPnts);
    data->loFreq = loFreq;
}

Ft::Ft(const Ft &rhs) : data(rhs.data)
{

}

Ft &Ft::operator=(const Ft &rhs)
{
    if (this != &rhs)
        data.operator=(rhs.data);
    return *this;
}

Ft::~Ft()
{

}

void Ft::setPoint(int i, QPointF pt, double ignoreRange)
{
    if(i >= 0 && i < data->ftData.size())
    {
        data->ftData[i] = pt;
        if(qAbs(pt.x()-data->loFreq) > ignoreRange)
        {
            data->yMin = qMin(pt.y(),data->yMin);
            data->yMax = qMax(pt.y(),data->yMax);
        }
    }
}

QPointF &Ft::operator[](int i)
{
    return data->ftData[i];
}

int Ft::size() const
{
    return data->ftData.size();
}

bool Ft::isEmpty() const
{
    return data->ftData.isEmpty();
}

QPointF Ft::at(int i) const
{
    if(i >= 0 && i < data->ftData.size())
        return data->ftData.at(i);

    return QPointF();
}

double Ft::xSpacing() const
{
    auto p0 = at(0);
    auto p1 = at(1);

    if(p1.isNull())
        return -1.0;

    return qAbs(p0.x() - p1.x());
}

double Ft::minFreq() const
{
    if(isEmpty())
        return -1;

    auto p0 = data->ftData.first();
    auto p1 = data->ftData.last();

    return qMin(p0.x(),p1.x());
}

double Ft::maxFreq() const
{
    if(isEmpty())
        return -1;

    auto p0 = data->ftData.first();
    auto p1 = data->ftData.last();

    return qMax(p0.x(),p1.x());
}

double Ft::loFreq() const
{
    return data->loFreq;
}

double Ft::yMin() const
{
    return data->yMin;
}

double Ft::yMax() const
{
    return data->yMax;
}

