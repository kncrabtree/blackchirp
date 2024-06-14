#include <modules/lif/data/liftrace.h>
#include <QtEndian>

#include <data/analysis/analysis.h>

LifTrace::LifTrace() : p_data(new LifTraceData)
{
}

LifTrace::LifTrace(const LifDigitizerConfig &c, const QVector<qint8> b, int dIndex, int lIndex, int bitShift)
    : p_data(new LifTraceData)
{
    //reference channel is used to normalize to pulse energy
    //if active, must be second channel
    p_data->xSpacing = c.xIncr();
    p_data->lifYMult = c.yMult(c.d_lifChannel)/(1 << bitShift);
    p_data->refYMult = c.yMult(c.d_refChannel)/(1 << bitShift);
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
            auto y = b.at(i);
            dat = static_cast<qint64>(y) << bitShift;
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
            dat = static_cast<qint64>(y) << bitShift;
        }
        p_data->lifData[i/incr] = dat;
    }
    if(c.d_refEnabled)
    {
        p_data->refData.resize(c.d_recordLength);
        qint64 dat = 0;
        for(int i=c.d_bytesPerPoint*refoffset; i<refoffset+incr*c.d_recordLength; i+=incr)
        {
            if(c.d_bytesPerPoint == 1)
            {
                char y = b.at(i);
                dat = static_cast<qint64>(y) << bitShift;
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
                dat = static_cast<qint64>(y) << bitShift;
            }
            p_data->refData[(i-c.d_bytesPerPoint*refoffset)/incr] = dat;
        }
    }
    p_data->shots = c.d_numAverages;
}

LifTrace::LifTrace(int di, int li, QVector<qint64> ld, QVector<qint64> rd, int shots, double xsp, double lym, double rym) : p_data(new LifTraceData)
{
    p_data->delayIndex = di;
    p_data->laserIndex = li;
    p_data->lifData = ld;
    p_data->refData = rd;
    p_data->shots = shots;
    p_data->xSpacing = xsp;
    p_data->lifYMult = lym;
    p_data->refYMult = rym;
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

double LifTrace::integrate(const LifProcSettings &s) const
{
    //validate ranges (sort of; if ranges are bad this will return 0);
    //lif start must be in range of data
    if(p_data->lifData.size() < 2)
        return 0.0;

    auto ls = qBound(0,s.lifGateStart,p_data->lifData.size()-2);
    //lif end must be greater than start and in range of data
    auto le = qBound(ls+1,s.lifGateEnd,p_data->lifData.size()-1);

    //do trapezoidal integration in integer/point space.
    //each segment has a width of 1 unit, and the area is (y_i + y_{i+1})/2
    //(think of it as end points have weight of 1, middle points have weight 2)
    //add up all y_i + y_{i+1}, then divide by 2 at the end

    auto l = lifToXY(s);


    double sum = 0.0;
    for(int i = ls; i<le-1; i++)
        sum += l.at(i).y() + l.at(i+1).y();

    auto lifInt = sum/2.0;

    //if no reference; just return raw integral
    if(!hasRefData())
        return lifInt;

    auto r = refToXY(s);

    auto rs = qBound(0,s.refGateStart,p_data->refData.size()-2);
    auto re = qBound(rs+1,s.refGateEnd,p_data->refData.size()-1);

    sum = 0.0;
    for(int i = rs; i<re-1; i++)
        sum += r.at(i).y() + r.at(i+1).y();

    double refInt = sum/2.0;

    //don't divide by 0!
    if(qFuzzyCompare(1.0,1.0+refInt))
        return lifInt;
    else
        return lifInt/refInt;
}

int LifTrace::delayIndex() const
{
    return p_data->delayIndex;
}

int LifTrace::laserIndex() const
{
    return p_data->laserIndex;
}

QVector<QPointF> LifTrace::lifToXY(const LifProcSettings &s) const
{
    QVector<double> y;
    y.resize(p_data->lifData.size());

    for(int i=0; i<p_data->lifData.size(); i++)
    {
        if(p_data->shots == 1)
            y[i] = static_cast<double>(p_data->lifData.at(i))*p_data->lifYMult;
        else
            y[i] = (static_cast<double>(p_data->lifData.at(i))*p_data->lifYMult/static_cast<double>(p_data->shots));
    }

    return processXY(y,s);
}

QVector<QPointF> LifTrace::refToXY(const LifProcSettings &s) const
{
    QVector<double> y;
    y.resize(p_data->refData.size());

    for(int i=0; i<p_data->refData.size(); i++)
    {
        if(p_data->shots == 1)
            y[i] = static_cast<double>(p_data->refData.at(i))*p_data->refYMult;
        else
            y[i] = (static_cast<double>(p_data->refData.at(i))*p_data->refYMult/static_cast<double>(p_data->shots));
    }

    return processXY(y,s);
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

double LifTrace::lifYMult() const
{
    return p_data->lifYMult;
}

double LifTrace::refYMult() const
{
    return p_data->refYMult;
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

QVector<QPointF> LifTrace::processXY(const QVector<double> d, const LifProcSettings &s) const
{
    //low-pass first, then Sav-Gol
    auto ynew = d;
    if(s.lowPassAlpha > 1e-5)
    {
        for(int i=1; i<ynew.size(); i++)
            ynew[i] = s.lowPassAlpha*ynew.at(i-1) + (1-s.lowPassAlpha)*ynew.at(i);
    }

    if(s.savGolEnabled)
    {
        auto coefs = Analysis::calcSavGolCoefs(s.savGolWin,s.savGolPoly);
        ynew = Analysis::savGolSmooth(coefs,0,ynew,xSpacingns());
    }

    QVector<QPointF> out;
    out.resize(ynew.size());
    for(int i=0; i<ynew.size(); i++)
        out[i] = {xSpacingns()*i,ynew[i]};

    return out;
}
