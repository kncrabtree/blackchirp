#include <modules/lif/data/liftrace.h>
#include <QtEndian>

#include <data/analysis/analysis.h>

LifTrace::LifTrace() : p_data(new LifTraceData)
{
}

LifTrace::LifTrace(const LifDigitizerConfig &c, const QByteArray b, int dIndex, int lIndex)
    : p_data(new LifTraceData)
{
    //reference channel is used to normalize to pulse energy
    //if active, must be second channel
    p_data->xSpacing = c.xIncr();
    p_data->lifYMult = c.yMult(c.d_lifChannel);
    p_data->refYMult = c.yMult(c.d_refChannel);
    p_data->delayIndex = dIndex;
    p_data->laserIndex = lIndex;

    p_data->lifData.resize(c.d_recordLength);
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
        p_data->lifData[i/incr] = dat;
    }
    if(c.d_refEnabled)
    {
        p_data->refData.resize(c.d_recordLength);
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
            p_data->refData[(i-c.d_bytesPerPoint*refoffset)/incr] = dat;
        }
    }
    p_data->shots = c.d_numAverages;
}

LifTrace::LifTrace(const LifTrace &other) : p_data(other.p_data)
{
}

LifTrace &LifTrace::operator=(const LifTrace &other)
{
    if (this != &other)
        p_data.operator=(other.p_data);
    return *this;
}

double LifTrace::integrate(int gl1, int gl2, int gr1, int gr2) const
{
    if(gl1 < 0)
        gl1 = 0;
    if(gl2 < 0)
        gl2 = p_data->lifData.size()-1;

    //validate ranges (sort of; if ranges are bad this will return 0);
    //lif start must be in range of data
    gl1 = qBound(0,gl1,p_data->lifData.size());
    //lif end must be greater than start and in range of data
    gl2 = qBound(gl1,gl2,p_data->lifData.size());
    //lif start must be less than end
    gl1 = qBound(0,gl1,gl2);

    //do trapezoidal integration in integer/point space.
    //each segment has a width of 1 unit, and the area is (y_i + y_{i+1})/2
    //(think of it as end points have weight of 1, middle points have weight 2)
    //add up all y_i + y_{i+1}, then divide by 2 at the end

    qint64 sum = 0;
    for(int i = gl1; i<gl2-1; i++)
        sum += static_cast<qint64>(p_data->lifData.at(i)) + static_cast<qint64>(p_data->lifData.at(i+1));

    //multiply by y spacing and x spacing
    double out = static_cast<double>(sum)/2.0*p_data->lifYMult*p_data->xSpacing;

    if(p_data->shots > 1)
        out /= static_cast<double>(p_data->shots);

    //if no reference; just return raw integral
    if(p_data->refData.size() == 0)
        return out;

    if(gr1 < 0)
        gr1 = 0;
    if(gr2 < 0)
        gr2 = p_data->refData.size()-1;

    sum = 0;
    for(int i = gr1; i<gr2-1; i++)
        sum += static_cast<qint64>(p_data->refData.at(i)) + static_cast<qint64>(p_data->refData.at(i+1));

    double ref = static_cast<double>(sum)/2.0*p_data->refYMult*p_data->xSpacing;

    if(p_data->shots > 1)
        ref /= static_cast<double>(p_data->shots);

    //don't divide by 0!
    if(qFuzzyCompare(1.0,1.0+ref))
        return out;
    else
        return out/ref;
}

int LifTrace::delayIndex() const
{
    return p_data->delayIndex;
}

int LifTrace::laserIndex() const
{
    return p_data->laserIndex;
}

QVector<QPointF> LifTrace::lifToXY() const
{
    QVector<QPointF> out;
    out.resize(p_data->lifData.size());

    for(int i=0; i<p_data->lifData.size(); i++)
    {
        out[i].setX(static_cast<double>(i)*p_data->xSpacing*1e9); //convert to ns
        if(p_data->shots == 1)
            out[i].setY(static_cast<double>(p_data->lifData.at(i))*p_data->lifYMult);
        else
            out[i].setY(static_cast<double>(p_data->lifData.at(i))*p_data->lifYMult/static_cast<double>(p_data->shots));
    }

    return out;
}

QVector<QPointF> LifTrace::refToXY() const
{
    QVector<QPointF> out;
    out.resize(p_data->refData.size());

    for(int i=0; i<p_data->refData.size(); i++)
    {
        out[i].setX(static_cast<double>(i)*p_data->xSpacing*1e9); // convert to ns
        if(p_data->shots == 1)
            out[i].setY(static_cast<double>(p_data->refData.at(i))*p_data->refYMult);
        else
            out[i].setY(static_cast<double>(p_data->refData.at(i))*p_data->refYMult/static_cast<double>(p_data->shots));
    }

    return out;
}

double LifTrace::maxTime() const
{
    if(p_data->lifData.isEmpty())
        return 0.0;

    return static_cast<double>(p_data->lifData.size()-1.0)*p_data->xSpacing;
}

QVector<qint64> LifTrace::lifRaw() const
{
    return p_data->lifData;
}

QVector<qint64> LifTrace::refRaw() const
{
    return p_data->refData;
}

int LifTrace::shots() const
{
    return p_data->shots;
}

int LifTrace::size() const
{
    return p_data->lifData.size();
}

bool LifTrace::hasRefData() const
{
    return !p_data->refData.isEmpty();
}

double LifTrace::xSpacing() const
{
    return p_data->xSpacing;
}

double LifTrace::xSpacingns() const
{
    return p_data->xSpacing*1e9;
}

void LifTrace::add(const LifTrace &other)
{
    if(other.size() != size())
        return;

    auto l = other.lifRaw();
    auto r = other.refRaw();

    for(int i=0; i<p_data->lifData.size(); i++)
        p_data->lifData[i] += l.at(i);

    for(int i=0; i<p_data->refData.size(); i++)
        p_data->refData[i] += r.at(i);

    p_data->shots += other.shots();
}

void LifTrace::rollAvg(const LifTrace &other, int numShots)
{

    auto c = shots();
    if(c + other.shots() <= numShots)
        add(other);
    else
    {
        auto l = other.lifRaw();
        auto r = other.refRaw();

        for(int i=0; i<p_data->lifData.size(); i++)
            p_data->lifData[i] = Analysis::intRoundClosest(numShots*(p_data->lifData.at(i)+l.at(i)),numShots+1);

        for(int i=0; i<p_data->refData.size(); i++)
            p_data->refData[i] = Analysis::intRoundClosest(numShots*(p_data->refData.at(i)+r.at(i)),numShots+1);

        p_data->shots = numShots;
    }
}
