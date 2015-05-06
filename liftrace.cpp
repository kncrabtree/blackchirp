#include "liftrace.h"

LifTrace::LifTrace() : data(new LifTraceData)
{
}

LifTrace::LifTrace(const BlackChirp::LifScopeConfig c, const QByteArray b) : data(new LifTraceData)
{
    //reference channel is used to normalize to pulse energy
    //if active, must be second channel
    data->xSpacing = c.xIncr;
    data->lifYMult = c.yMult1;
    data->refYMult = c.yMult2;
    data->count = 1;

    data->lifData.resize(c.recordLength);
    for(int i=0; i<c.recordLength; i++)
    {
        if(c.bytesPerPoint == 1)
            data->lifData[i] = static_cast<qint16>(b.at(i));
        else
        {
            qint16 dat;
            if(c.byteOrder == QDataStream::LittleEndian)
                dat = (b.at(2*i + 1) << 8) | (b.at(2*i) & 0xff);
            else
                dat = (b.at(2*i) << 8) | (b.at(2*i + 1) & 0xff);
            data->lifData[i] = dat;
        }
    }

    if(c.refEnabled)
    {
        data->refData.resize(c.recordLength);
        for(int i=0; i<c.recordLength; i++)
        {
            if(c.bytesPerPoint == 1)
                data->refData[i] = static_cast<qint16>(b.at(c.recordLength + i));
            else
            {
                qint16 dat;
                if(c.byteOrder == QDataStream::LittleEndian)
                    dat = (b.at(2*(i+c.recordLength) + 1) << 8) | (b.at(2*(i+c.recordLength)) & 0xff);
                else
                    dat = (b.at(2*(i+c.recordLength)) << 8) | (b.at(2*(i+c.recordLength) + 1) & 0xff);
                data->refData[i] = dat;
            }
        }
    }
}

LifTrace::LifTrace(const LifTrace &rhs) : data(rhs.data)
{

}

LifTrace &LifTrace::operator=(const LifTrace &rhs)
{
    if (this != &rhs)
        data.operator=(rhs.data);
    return *this;
}

LifTrace::~LifTrace()
{

}

double LifTrace::integrate(int gl1, int gl2, int gr1, int gr2) const
{
    if(gl1 < 0)
        gl1 = 0;
    if(gl2 < 0)
        gl2 = data->lifData.size()-1;

    //validate ranges (sort of; if ranges are bad this will return 0);
    //lif start must be in range of data
    gl1 = qBound(0,gl1,data->lifData.size());
    //lif end must be greater than start and in range of data
    gl2 = qBound(gl1,gl2,data->lifData.size());
    //lif start must be less than end
    gl1 = qBound(0,gl1,gl2);

    //do trapezoidal integration in integer/point space.
    //each segment has a width of 1 unit, and the area is (y_i + y_{i+1})/2
    //(think of it as end points have weight of 1, middle points have weight 2)
    //add up all y_i + y_{i+1}, then divide by 2 at the end

    qint64 sum = 0;
    for(int i = gl1; i<gl2-1; i++)
        sum += static_cast<qint64>(data->lifData.at(i)) + static_cast<qint64>(data->lifData.at(i+1));

    double out = static_cast<double>(sum)/2.0*data->lifYMult;

    if(data->count > 1)
        out /= static_cast<double>(data->count);

    //if no reference; just return raw integral
    if(data->refData.size() == 0)
        return out;

    if(gr1 < 0)
        gr1 = 0;
    if(gr2 < 0)
        gr2 = data->refData.size()-1;

    sum = 0;
    for(int i = gr1; i<gr2-1; i++)
        sum += static_cast<qint64>(data->refData.at(i)) + static_cast<qint64>(data->refData.at(i+1));

    double ref = static_cast<double>(sum)/2.0*data->refYMult;

    if(data->count > 1)
        ref /= static_cast<double>(data->count);

    //don't divide by 0!
    if(qFuzzyCompare(1.0,1.0+ref))
        return out;
    else
        return out/ref;
}

double LifTrace::spacing() const
{
    return data->xSpacing;
}

QVector<QPointF> LifTrace::lifToXY() const
{
    QVector<QPointF> out;
    out.resize(data->lifData.size());

    for(int i=0; i<data->lifData.size(); i++)
    {
        out[i].setX(static_cast<double>(i)*data->xSpacing);
        if(data->count == 1)
            out[i].setY(static_cast<double>(data->lifData.at(i))*data->lifYMult);
        else
            out[i].setY(static_cast<double>(data->lifData.at(i))*data->lifYMult/static_cast<double>(data->count));
    }

    return out;
}

QVector<QPointF> LifTrace::refToXY() const
{
    QVector<QPointF> out;
    out.resize(data->refData.size());

    for(int i=0; i<data->refData.size(); i++)
    {
        out[i].setX(static_cast<double>(i)*data->xSpacing);
        if(data->count == 1)
            out[i].setY(static_cast<double>(data->refData.at(i))*data->refYMult);
        else
            out[i].setY(static_cast<double>(data->refData.at(i))*data->refYMult/static_cast<double>(data->count));
    }

    return out;
}

double LifTrace::maxTime() const
{
    if(data->lifData.isEmpty())
        return 0.0;

    return static_cast<double>(data->lifData.size()-1.0)*data->xSpacing;
}

qint64 LifTrace::lifAtRaw(int i) const
{
    return data->lifData.at(i);
}

qint64 LifTrace::refAtRaw(int i) const
{
    return data->refData.at(i);
}

int LifTrace::count() const
{
    return data->count;
}

int LifTrace::size() const
{
    return data->lifData.size();
}

bool LifTrace::hasRefData() const
{
    return !data->refData.isEmpty();
}

void LifTrace::add(const LifTrace &other)
{
    for(int i=0; i<data->lifData.size(); i++)
        data->lifData[i] += other.lifAtRaw(i);

    for(int i=0; i<data->refData.size(); i++)
        data->refData[i] += other.refAtRaw(i);

    data->count += other.count();
}

void LifTrace::rollAvg(const LifTrace &other, int numShots)
{
    if(data->count + other.count() <= numShots)
        add(other);
    else
    {
        for(int i=0; i<data->lifData.size(); i++)
            data->lifData[i] = (static_cast<double>(numShots)*(other.lifAtRaw(i) + data->lifData.at(i)))
                    /static_cast<double>(data->count + other.count());

        for(int i=0; i<data->refData.size(); i++)
            data->refData[i] = (static_cast<double>(numShots)*(other.refAtRaw(i) + data->refData.at(i)))
                    /static_cast<double>(data->count + other.count());

        data->count = numShots;
    }
}

