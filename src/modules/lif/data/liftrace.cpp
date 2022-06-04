#include <modules/lif/data/liftrace.h>
#include <QtEndian>

#include <data/analysis/analysis.h>

LifTrace::LifTrace()
{
}

LifTrace::LifTrace(const LifDigitizerConfig &c, const QByteArray b)
{
    //reference channel is used to normalize to pulse energy
    //if active, must be second channel
    d_xSpacing = c.xIncr();
    d_lifYMult = c.yMult(c.d_lifChannel);
    d_refYMult = c.yMult(c.d_refChannel);

    d_lifData.resize(c.d_recordLength);
    int incr = c.d_bytesPerPoint;
    int refoffset = c.d_recordLength;
    if(c.d_refEnabled && (c.d_channelOrder == LifDigitizerConfig::Interleaved))
    {
        incr = 2*c.d_bytesPerPoint;
        refoffset = c.d_bytesPerPoint;
    }

    for(int i=0; i<incr*c.d_recordLength; i+=incr)
    {
        qint64 dat = 0;
        if(c.d_bytesPerPoint == 1)
        {
            char y = b.at(i);
            dat = static_cast<qint64>(y);
        }
        else
        {
            auto y1 = static_cast<quint8>(b.at(2*i));
            auto y2 = static_cast<quint8>(b.at(2*i + 1));
            qint16 y = 0;
            y |= y1;
            y |= (y2 << 8);
            if(c.d_byteOrder == DigitizerConfig::BigEndian)
                y = qFromBigEndian(y);
            else
                y = qFromLittleEndian(y);
            dat = static_cast<qint64>(y);
        }
        d_lifData[i/incr] = dat;
    }
    if(c.d_refEnabled)
    {
        d_refData.resize(c.d_recordLength);
        qint64 dat = 0;
        for(int i=c.d_bytesPerPoint*refoffset; i<incr*c.d_recordLength; i+=incr)
        {
            if(c.d_bytesPerPoint == 1)
            {
                char y = b.at(i);
                dat = static_cast<qint64>(y);
            }
            else
            {
                auto y1 = static_cast<quint8>(b.at(2*i));
                auto y2 = static_cast<quint8>(b.at(2*i + 1));
                qint16 y = 0;
                y |= y1;
                y |= (y2 << 8);
                if(c.d_byteOrder == DigitizerConfig::BigEndian)
                    y = qFromBigEndian(y);
                else
                    y = qFromLittleEndian(y);
                dat = static_cast<qint64>(y);
            }
            d_refData[(i-c.d_bytesPerPoint*refoffset)/incr] = dat;
        }
    }
}

LifTrace::LifTrace(double lm, double rm, double sp, int count, const QVector<qint64> l, const QVector<qint64> r)
{
    d_lifYMult = lm;
    d_refYMult = rm;
    d_xSpacing = sp;
    d_count = count;
    d_lifData = l;
    d_refData = r;
}

double LifTrace::integrate(int gl1, int gl2, int gr1, int gr2) const
{
    if(gl1 < 0)
        gl1 = 0;
    if(gl2 < 0)
        gl2 = d_lifData.size()-1;

    //validate ranges (sort of; if ranges are bad this will return 0);
    //lif start must be in range of data
    gl1 = qBound(0,gl1,d_lifData.size());
    //lif end must be greater than start and in range of data
    gl2 = qBound(gl1,gl2,d_lifData.size());
    //lif start must be less than end
    gl1 = qBound(0,gl1,gl2);

    //do trapezoidal integration in integer/point space.
    //each segment has a width of 1 unit, and the area is (y_i + y_{i+1})/2
    //(think of it as end points have weight of 1, middle points have weight 2)
    //add up all y_i + y_{i+1}, then divide by 2 at the end

    qint64 sum = 0;
    for(int i = gl1; i<gl2-1; i++)
        sum += static_cast<qint64>(d_lifData.at(i)) + static_cast<qint64>(d_lifData.at(i+1));

    //multiply by y spacing and x spacing
    double out = static_cast<double>(sum)/2.0*d_lifYMult*d_xSpacing;

    if(d_count > 1)
        out /= static_cast<double>(d_count);

    //if no reference; just return raw integral
    if(d_refData.size() == 0)
        return out;

    if(gr1 < 0)
        gr1 = 0;
    if(gr2 < 0)
        gr2 = d_refData.size()-1;

    sum = 0;
    for(int i = gr1; i<gr2-1; i++)
        sum += static_cast<qint64>(d_refData.at(i)) + static_cast<qint64>(d_refData.at(i+1));

    double ref = static_cast<double>(sum)/2.0*d_refYMult*d_xSpacing;

    if(d_count > 1)
        ref /= static_cast<double>(d_count);

    //don't divide by 0!
    if(qFuzzyCompare(1.0,1.0+ref))
        return out;
    else
        return out/ref;
}

QVector<QPointF> LifTrace::lifToXY() const
{
    QVector<QPointF> out;
    out.resize(d_lifData.size());

    for(int i=0; i<d_lifData.size(); i++)
    {
        out[i].setX(static_cast<double>(i)*d_xSpacing*1e9); //convert to ns
        if(d_count == 1)
            out[i].setY(static_cast<double>(d_lifData.at(i))*d_lifYMult);
        else
            out[i].setY(static_cast<double>(d_lifData.at(i))*d_lifYMult/static_cast<double>(d_count));
    }

    return out;
}

QVector<QPointF> LifTrace::refToXY() const
{
    QVector<QPointF> out;
    out.resize(d_refData.size());

    for(int i=0; i<d_refData.size(); i++)
    {
        out[i].setX(static_cast<double>(i)*d_xSpacing*1e9); // convert to ns
        if(d_count == 1)
            out[i].setY(static_cast<double>(d_refData.at(i))*d_refYMult);
        else
            out[i].setY(static_cast<double>(d_refData.at(i))*d_refYMult/static_cast<double>(d_count));
    }

    return out;
}

double LifTrace::maxTime() const
{
    if(d_lifData.isEmpty())
        return 0.0;

    return static_cast<double>(d_lifData.size()-1.0)*d_xSpacing;
}

QVector<qint64> LifTrace::lifRaw() const
{
    return d_lifData;
}

QVector<qint64> LifTrace::refRaw() const
{
    return d_refData;
}

qint64 LifTrace::lifAtRaw(int i) const
{
    return d_lifData.at(i);
}

qint64 LifTrace::refAtRaw(int i) const
{
    return d_refData.at(i);
}

int LifTrace::count() const
{
    return d_count;
}

int LifTrace::size() const
{
    return d_lifData.size();
}

bool LifTrace::hasRefData() const
{
    return !d_refData.isEmpty();
}

void LifTrace::add(const LifTrace &other)
{
    for(int i=0; i<d_lifData.size(); i++)
        d_lifData[i] += other.lifAtRaw(i);

    for(int i=0; i<d_refData.size(); i++)
        d_refData[i] += other.refAtRaw(i);

    d_count += other.count();
}

void LifTrace::rollAvg(const LifTrace &other, int numShots)
{
    if(d_count + other.count() <= numShots)
        add(other);
    else
    {
        for(int i=0; i<d_lifData.size(); i++)
            d_lifData[i] = Analysis::intRoundClosest(numShots*(lifAtRaw(i)+other.lifAtRaw(i)),numShots+1);

        for(int i=0; i<d_refData.size(); i++)
            d_refData[i] = Analysis::intRoundClosest(numShots*(refAtRaw(i)+other.refAtRaw(i)),numShots+1);

        d_count = numShots;
    }
}
