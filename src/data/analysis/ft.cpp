#include <src/data/analysis/ft.h>

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

void Ft::resize(int n, double ignoreRange)
{
    data->ftData.resize(n);
    data->yMin = 0.0;
    data->yMax = 0.0;
    for(int i=0; i<data->ftData.size(); i++)
    {
        if(qAbs(data->ftData.at(i).x()-data->loFreq) > ignoreRange)
        {
            data->yMin = qMin(data->ftData.at(i).y(),data->yMin);
            data->yMax = qMax(data->ftData.at(i).y(),data->yMax);
        }
    }
}

QPointF &Ft::operator[](int i)
{
    return data->ftData[i];
}

void Ft::reserve(int n)
{
    data->ftData.reserve(n);
}

void Ft::append(QPointF pt, double ignoreRange)
{
    data->ftData.append(pt);
    if(qAbs(pt.x()-data->loFreq) > ignoreRange)
    {
        data->yMax = qMax(data->yMax,pt.y());
        data->yMin = qMin(data->yMin,pt.y());
    }
}

void Ft::trim(double minOffset, double maxOffset)
{
    QVector<QPointF> newData;
    newData.reserve(data->ftData.size());
    data->yMin = 0.0;
    data->yMax = 0.0;
    bool reverse = false;
    if(data->ftData.constLast().x() < data->ftData.constFirst().x())
        reverse = true;
    for(int i=0; i<data->ftData.size(); i++)
    {
        if(reverse)
            i = data->ftData.size()-1-i;

        double thisOffset = qAbs(data->ftData.at(i).x() - data->loFreq);
        if(thisOffset < minOffset || thisOffset > maxOffset)
            continue;

        if(reverse)
            newData.prepend(data->ftData.at(i));
        else
            newData.append(data->ftData.at(i));

        data->yMax = qMax(data->yMax,data->ftData.at(i).y());
        data->yMin = qMin(data->yMin,data->ftData.at(i).y());
    }

    data->ftData = newData;

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
//    if(i >= 0 && i < data->ftData.size())
    return data->ftData.at(i);

    //    return QPointF();
}

QPointF Ft::constFirst() const
{
    return data->ftData.constFirst();
}

QPointF Ft::constLast() const
{
    return data->ftData.constLast();
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

    auto p0 = data->ftData.constFirst();
    auto p1 = data->ftData.constLast();

    return qMin(p0.x(),p1.x());
}

double Ft::maxFreq() const
{
    if(isEmpty())
        return -1;

    auto p0 = data->ftData.constFirst();
    auto p1 = data->ftData.constLast();

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

QVector<double> Ft::xData() const
{
    QVector<double> out;
    out.resize(data->ftData.size());
    for(int i=0; i<data->ftData.size(); i++)
        out[i] = data->ftData.at(i).x();

    return out;
}

QVector<double> Ft::yData() const
{
    QVector<double> out;
    out.resize(data->ftData.size());
    for(int i=0; i<data->ftData.size(); i++)
        out[i] = data->ftData.at(i).y();

    return out;
}

QVector<QPointF> Ft::toVector() const
{
    return data->ftData;
}

