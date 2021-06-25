#include <modules/lif/data/liftrace.h>
#include <QtEndian>

#include <data/analysis/analysis.h>

LifTrace::LifTrace() : data(new LifTraceData)
{
}
LifTrace::LifTrace(const BlackChirp::LifScopeConfig &c, const QByteArray b) : data(new LifTraceData)
{
    //reference channel is used to normalize to pulse energy
    //if active, must be second channel
    data->xSpacing = c.xIncr;
    data->lifYMult = c.yMult1;
    data->refYMult = c.yMult2;
    data->count = 1;

    data->lifData.resize(c.recordLength);
    int incr = 1;
    int refoffset = c.recordLength;
    if(c.refEnabled && (c.channelOrder == BlackChirp::ChannelsInterleaved))
    {
        incr = 2;
        refoffset = 1;
    }

    for(int i=0; i<incr*c.recordLength; i+=incr)
    {
        qint64 dat = 0;
        if(c.bytesPerPoint == 1)
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
            if(c.byteOrder == QDataStream::BigEndian)
                y = qFromBigEndian(y);
            else
                y = qFromLittleEndian(y);
            dat = static_cast<qint64>(y);
        }
        data->lifData[i/incr] = dat;
    }
    if(c.refEnabled)
    {
        data->refData.resize(c.recordLength);
        qint64 dat = 0;
        for(int i=c.bytesPerPoint*refoffset; i<incr*c.recordLength; i+=incr)
        {
            if(c.bytesPerPoint == 1)
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
                if(c.byteOrder == QDataStream::BigEndian)
                    y = qFromBigEndian(y);
                else
                    y = qFromLittleEndian(y);
                dat = static_cast<qint64>(y);
            }
            data->refData[(i-c.bytesPerPoint*refoffset)/incr] = dat;
        }
    }
}

LifTrace::LifTrace(double lm, double rm, double sp, int count, const QVector<qint64> l, const QVector<qint64> r) :
    data(new LifTraceData)
{
    data->lifYMult = lm;
    data->refYMult = rm;
    data->xSpacing = sp;
    data->count = count;
    data->lifData = l;
    data->refData = r;
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

    //multiply by y spacing and x spacing
    double out = static_cast<double>(sum)/2.0*data->lifYMult*data->xSpacing;

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

    double ref = static_cast<double>(sum)/2.0*data->refYMult*data->xSpacing;

    if(data->count > 1)
        ref /= static_cast<double>(data->count);

    //don't divide by 0!
    if(qFuzzyCompare(1.0,1.0+ref))
        return out;
    else
        return out/ref;
}

double LifTrace::lifYMult() const
{
    return data->lifYMult;
}

double LifTrace::refYMult() const
{
    return data->refYMult;
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
        out[i].setX(static_cast<double>(i)*data->xSpacing*1e9); //convert to ns
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
        out[i].setX(static_cast<double>(i)*data->xSpacing*1e9); // convert to ns
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

QVector<qint64> LifTrace::lifRaw() const
{
    return data->lifData;
}

QVector<qint64> LifTrace::refRaw() const
{
    return data->refData;
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
            data->lifData[i] = Analysis::intRoundClosest(numShots*(lifAtRaw(i)+other.lifAtRaw(i)),numShots+1);

        for(int i=0; i<data->refData.size(); i++)
            data->refData[i] = Analysis::intRoundClosest(numShots*(refAtRaw(i)+other.refAtRaw(i)),numShots+1);

        data->count = numShots;
    }
}


QDataStream &operator<<(QDataStream &stream, const LifTrace &t)
{
    stream << t.lifYMult() << t.refYMult() << t.spacing() << t.count() << t.lifRaw() << t.refRaw();
    return stream;
}

QDataStream &operator>>(QDataStream &stream, LifTrace &t)
{
    double ly, ry, x;
    int c;
    QVector<qint64> l,r;

    stream >> ly >> ry >> x >> c >> l >> r;

    t = LifTrace(ly,ry,x,c,l,r);
    return stream;
}
