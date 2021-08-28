#include <data/analysis/ft.h>

class FtData : public QSharedData
{
public:
    FtData() {}

    double yMin{0.0};
    double yMax{0.0};
    double x0MHz{0.0};
    double spacingMHz{1.0};
    double loFreqMHz{0.0};

    QVector<double> ftData;
    quint64 shots{0};
};

Ft::Ft() : data(new FtData)
{

}

Ft::Ft(int numPnts, double f0, double spacing, double loFreq) : data(new FtData)
{
    data->ftData.resize(numPnts);
    data->x0MHz = f0;
    data->spacingMHz = spacing;
    data->loFreqMHz = loFreq;
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

void Ft::setPoint(int i, double y, double ignoreRange)
{
    if(i >= 0 && i < data->ftData.size())
    {
        data->ftData[i] = y;
        if(qAbs(xAt(i)-data->loFreqMHz) > ignoreRange)
        {
            data->yMin = qMin(y,data->yMin);
            data->yMax = qMax(y,data->yMax);
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
        if(qAbs(xAt(i)-data->loFreqMHz) > ignoreRange)
        {
            data->yMin = qMin(at(i),data->yMin);
            data->yMax = qMax(at(i),data->yMax);
        }
    }
}

double &Ft::operator[](int i)
{
    return data->ftData[i];
}

void Ft::reserve(int n)
{
    data->ftData.reserve(n);
}

void Ft::squeeze()
{
    data->ftData.squeeze();
}

void Ft::setLoFreq(double f)
{
    data->loFreqMHz = f;
}

void Ft::setX0(double d)
{
    data->x0MHz = d;
}

void Ft::setSpacing(double s)
{
    data->spacingMHz = s;
}

void Ft::append(double y)
{
    data->ftData.append(y);
    data->yMax = qMax(data->yMax,y);
    data->yMin = qMin(data->yMin,y);
}

void Ft::trim(double minOffset, double maxOffset)
{
    //assume the FT could contain the upper or lower sideband or both
    double usbmin = data->loFreqMHz + minOffset;
    double usbmax = data->loFreqMHz + maxOffset;
    double lsbmin = data->loFreqMHz - minOffset;
    double lsbmax = data->loFreqMHz - maxOffset;

    //get indices of first and last points
    int minlIndex = static_cast<int>((lsbmin-data->x0MHz)/data->spacingMHz);
    int maxlIndex = static_cast<int>((lsbmax-data->x0MHz)/data->spacingMHz);
    int minuIndex = static_cast<int>((usbmin-data->x0MHz)/data->spacingMHz);
    int maxuIndex = static_cast<int>((usbmax-data->x0MHz)/data->spacingMHz);

    if(minlIndex > maxlIndex)
        qSwap(minlIndex,maxlIndex);
    if(minuIndex > maxuIndex)
        qSwap(minuIndex,maxuIndex);

    int minIndex = 0, maxIndex = size()-1;

    //these are the first and last points to retain
    minIndex = qMax(minIndex,qMax(minlIndex,minuIndex));
    maxIndex = qMin(maxIndex,qMin(maxlIndex,maxuIndex));

    //update f0 to match new reference frequency and update y limits
    data->x0MHz = xAt(minIndex);
    data->ftData = data->ftData.mid(minIndex,maxIndex-minIndex+1);
    data->yMin = 0.0;
    data->yMax = 0.0;
    for(int i=0; i<data->ftData.size(); i++)
    {
        data->yMax = qMax(data->yMax,at(i));
        data->yMin = qMin(data->yMin,at(i));
    }
}

void Ft::setNumShots(quint64 shots)
{
    data->shots = shots;
}

void Ft::setData(const QVector<double> d, double yMin, double yMax)
{
    data->ftData = d;
    data->yMin = yMin;
    data->yMax = yMax;
}

int Ft::size() const
{
    return data->ftData.size();
}

bool Ft::isEmpty() const
{
    return data->ftData.isEmpty();
}

double Ft::at(int i) const
{
    return data->ftData.at(i);
}

double Ft::value(int i) const
{
    return data->ftData.value(i,0.0);
}

double Ft::constFirst() const
{
    return data->ftData.constFirst();
}

double Ft::constLast() const
{
    return data->ftData.constLast();
}

double Ft::xAt(int i) const
{
    return data->x0MHz + (double)i*data->spacingMHz;
}

double Ft::xFirst() const
{
    return data->x0MHz;
}

double Ft::xLast() const
{
    return data->x0MHz + (double)(size()-1)*data->spacingMHz;
}

double Ft::xSpacing() const
{
    return data->spacingMHz;
}

double Ft::minFreqMHz() const
{
    return qMin(xFirst(),xLast());
}

double Ft::maxFreqMHz() const
{
    return qMax(xFirst(),xLast());
}

double Ft::loFreqMHz() const
{
    return data->loFreqMHz;
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
    auto s = data->ftData.size();
    out.reserve(s);
    for(int i=0; i<s; i++)
        out.append(xAt(i));

    return out;
}

QVector<double> Ft::yData() const
{
    return data->ftData;
}

QVector<QPointF> Ft::toVector() const
{
    QVector<QPointF> out;
    auto s = data->ftData.size();
    out.reserve(s);
    for(int i=0; i<s; i++)
        out.append({xAt(i),at(i)});
    return out;
}

quint64 Ft::shots() const
{
    return data->shots;
}

