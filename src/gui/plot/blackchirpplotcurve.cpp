#include "blackchirpplotcurve.h"

#include <QPalette>
#include <QMutex>

BlackchirpPlotCurveBase::BlackchirpPlotCurveBase(const QString key, const QString title, Qt::PenStyle defaultLineStyle, QwtSymbol::Style defaultMarker) :
    SettingsStorage({BC::Key::bcCurve,key},General), d_key{key},
    p_samplesMutex{new QMutex}
{
    if(!title.isEmpty())
        setTitle(title);
    else
        setTitle(key);

    setItemAttribute(QwtPlotItem::Legend);
    setItemAttribute(QwtPlotItem::AutoScale);
    setItemInterest(QwtPlotItem::ScaleInterest);

    getOrSetDefault(BC::Key::bcCurveStyle,static_cast<int>(defaultLineStyle));
    getOrSetDefault(BC::Key::bcCurveMarker,static_cast<int>(defaultMarker));

    configurePen();
    configureSymbol();
    setRenderHint(QwtPlotItem::RenderAntialiased);

    setAxes(get<QwtPlot::Axis>(BC::Key::bcCurveAxisX,QwtPlot::xBottom),
            get<QwtPlot::Axis>(BC::Key::bcCurveAxisY,QwtPlot::yLeft));
    setVisible(get<bool>(BC::Key::bcCurveVisible,true));

}

BlackchirpPlotCurveBase::~BlackchirpPlotCurveBase()
{
    delete p_samplesMutex;

}

void BlackchirpPlotCurveBase::setColor(const QColor c)
{
    set(BC::Key::bcCurveColor,c);
    configurePen();
    configureSymbol();
}

void BlackchirpPlotCurveBase::setLineThickness(double t)
{
    set(BC::Key::bcCurveThickness,t);
    configurePen();
}

void BlackchirpPlotCurveBase::setLineStyle(Qt::PenStyle s)
{
    set(BC::Key::bcCurveStyle,static_cast<int>(s));
    configurePen();
}

void BlackchirpPlotCurveBase::setMarkerStyle(QwtSymbol::Style s)
{
    set(BC::Key::bcCurveMarker,static_cast<int>(s));
    configureSymbol();
}

void BlackchirpPlotCurveBase::setMarkerSize(int s)
{
    set(BC::Key::bcCurveMarkerSize,s);
    configureSymbol();
}

void BlackchirpPlotCurveBase::setCurveVisible(bool v)
{
    set(BC::Key::bcCurveVisible,v);
    setVisible(v);
}

void BlackchirpPlotCurveBase::setCurveAxisX(QwtPlot::Axis a)
{
    set(BC::Key::bcCurveAxisX,static_cast<int>(a));
    setXAxis(a);
}

void BlackchirpPlotCurveBase::setCurveAxisY(QwtPlot::Axis a)
{
    set(BC::Key::bcCurveAxisY,static_cast<int>(a));
    setYAxis(a);
}

void BlackchirpPlotCurveBase::setCurvePlotIndex(int i)
{
    set(BC::Key::bcCurvePlotIndex,i);
}

void BlackchirpPlotCurveBase::updateFromSettings()
{
    configurePen();
    configureSymbol();
    setAxes(get<QwtPlot::Axis>(BC::Key::bcCurveAxisX,QwtPlot::xBottom),
            get<QwtPlot::Axis>(BC::Key::bcCurveAxisY,QwtPlot::yLeft));
    setVisible(get<bool>(BC::Key::bcCurveVisible,true));
}

void BlackchirpPlotCurveBase::configurePen()
{
    QPen p;
    QPalette pal;
    p.setColor(get<QColor>(BC::Key::bcCurveColor,pal.color(QPalette::BrightText)));
    p.setWidthF(get<double>(BC::Key::bcCurveThickness,1.0));
    p.setStyle(get<Qt::PenStyle>(BC::Key::bcCurveStyle,Qt::SolidLine));
    setPen(p);
}

void BlackchirpPlotCurveBase::configureSymbol()
{
    auto sym = new QwtSymbol();
    QPalette pal;
    sym->setStyle(get<QwtSymbol::Style>(BC::Key::bcCurveMarker,QwtSymbol::NoSymbol));
    sym->setColor(get<QColor>(BC::Key::bcCurveColor,pal.color(QPalette::BrightText)));
    sym->setPen(get<QColor>(BC::Key::bcCurveColor,pal.color(QPalette::BrightText)));
    auto s = get<int>(BC::Key::bcCurveMarkerSize,5);
    sym->setSize(QSize(s,s));
    setSymbol(sym);
}

void BlackchirpPlotCurveBase::setSamples(const QVector<QPointF> d)
{
    QMutexLocker l(p_samplesMutex);
    QwtPlotCurve::setSamples(d);

}

void BlackchirpPlotCurveBase::filter(int w, const QwtScaleMap map)
{
    setSamples(_filter(w,map));
}


void BlackchirpPlotCurveBase::draw(QPainter *painter, const QwtScaleMap &xMap, const QwtScaleMap &yMap, const QRectF &canvasRect) const
{
    QMutexLocker l(p_samplesMutex);
    QwtPlotSeriesItem::draw(painter,xMap,yMap,canvasRect);
}

BlackchirpPlotCurve::BlackchirpPlotCurve(const QString key, const QString title, Qt::PenStyle defaultLineStyle, QwtSymbol::Style defaultMarker) :
    BlackchirpPlotCurveBase(key,title,defaultLineStyle,defaultMarker), p_dataMutex{new QMutex}
{

}

BlackchirpPlotCurve::~BlackchirpPlotCurve()
{
    delete p_dataMutex;
}


void BlackchirpPlotCurve::setCurveData(const QVector<QPointF> d)
{
    p_dataMutex->lock();
    d_curveData = d;

    d_boundingRect = QRectF(1.0,1.0,-2.0,-2.0);

    if(!d.isEmpty())
    {
        d_boundingRect.setLeft(qMin(d.first().x(),d.last().x()));
        d_boundingRect.setRight(qMax(d.first().x(),d.last().x()));

        p_dataMutex->unlock();
        calcBoundingRectHeight();
    }
    p_dataMutex->unlock();
}

void BlackchirpPlotCurve::setCurveData(const QVector<QPointF> d, double min, double max)
{
    QMutexLocker l(p_dataMutex);
    d_curveData = d;

    if(!d.isEmpty())
    {
        d_boundingRect.setLeft(qMin(d.first().x(),d.last().x()));
        d_boundingRect.setRight(qMax(d.first().x(),d.last().x()));

        //according to the Qwt documentation, the bottom of the bounding rect is the max y value
        d_boundingRect.setBottom(max);
        d_boundingRect.setTop(min);
    }

}

void BlackchirpPlotCurve::appendPoint(const QPointF p)
{
    QMutexLocker l(p_dataMutex);
    d_curveData.append(p);
    if(d_curveData.size() == 1)
    {
        d_boundingRect.setTopLeft(p);
        d_boundingRect.setBottomRight(p);
    }
    else
    {
        d_boundingRect.setLeft(qMin(d_curveData.first().x(),d_curveData.last().x()));
        d_boundingRect.setRight(qMax(d_curveData.first().x(),d_curveData.last().x()));

        d_boundingRect.setBottom(qMax(d_boundingRect.bottom(),p.y()));
        d_boundingRect.setBottom(qMin(d_boundingRect.top(),p.y()));
    }
}

void BlackchirpPlotCurve::calcBoundingRectHeight()
{
    QMutexLocker l(p_dataMutex);
    auto top = d_curveData.first().y();
    auto bottom = d_curveData.first().y();

    for(auto p : d_curveData)
    {
        top = qMin(p.y(),top);
        bottom = qMax(p.y(),bottom);
    }

    d_boundingRect.setTop(top);
    d_boundingRect.setBottom(bottom);
}


QRectF BlackchirpPlotCurve::boundingRect() const
{

    QMutexLocker l(p_dataMutex);
    if(d_curveData.isEmpty())
        return QRectF(1.0,1.0,-2.0,-2.0);

    if((d_boundingRect.height() > 0.0) && (d_boundingRect.width() > 0.0))
        return d_boundingRect;


    auto left = qMin(d_curveData.first().x(),d_curveData.last().x());
    auto right = qMax(d_curveData.first().x(),d_curveData.last().x());

    auto top = d_curveData.first().y();
    auto bottom = d_curveData.first().y();

    for(auto &p : d_curveData)
    {
        top = qMin(p.y(),top);
        bottom = qMax(p.y(),bottom);
    }

    return QRectF( QPointF{left,top},QPointF{right,bottom} );
}

QVector<QPointF> BlackchirpPlotCurve::curveData() const
{
    return d_curveData;
}

QVector<QPointF> BlackchirpPlotCurve::_filter(int w, const QwtScaleMap map)
{
    p_dataMutex->lock();
    auto size = d_curveData.size();
    QVector<QPointF> d = d_curveData;
    d.detach();

    p_dataMutex->unlock();

    if(size < 2.5*w)
    {
        d_boundingRect = boundingRect();
        return d;
    }

    QVector<QPointF> filtered;
    filtered.reserve(2*w+2);

    int firstPixel = 0;
    int lastPixel = w;
    int inc = 1;

    auto firstx = d.constFirst().x();
    auto lastx = d.constLast().x();

    //XOR operation
    if(!(firstx > lastx) != !(map.invTransform(0) > map.invTransform(w)))
    {
        firstPixel = w;
        lastPixel = 0;
        inc = -1;
    }
    auto it = d.cbegin();

    //find first data point that is in the range of the plot
    while(it != d.cend() && (map.transform(it->x()) - static_cast<double>(firstPixel))*(double)inc < 0)
        ++it;

    //add the previous point to the filtered array
    //this will make sure the curve always goes to the edge of the plot
    if(it != d.cbegin())
    {
        --it;
        filtered.append({it->x(),it->y()});
        ++it;
    }

    //at this point, dataIndex is at the first point within the range of the plot. loop over pixels, compressing data
    for(int pixel = firstPixel; pixel!=lastPixel; pixel+=inc)
    {
        auto min = it->y();
        auto max = it->y();

        int numPnts = 0;
        double nextPixelX = map.invTransform(pixel+(double)inc);

        while(it != d.cend() && (it->x() - nextPixelX)*(double)inc < 0)
        {
            min = qMin(it->y(),min);
            max = qMax(it->y(),max);

            ++it;
            numPnts++;
        }


        if(numPnts == 1)
        {
            --it;
            filtered.append({it->x(),it->y()});
            ++it;
        }
        else if (numPnts > 1)
        {
            auto x = map.invTransform(pixel);
            filtered.append({x,min});
            filtered.append({x,max});
        }

    }

    if(it != d.cend())
        filtered.append({it->x(),it->y()});

    return filtered;
}

BCEvenSpacedCurveBase::BCEvenSpacedCurveBase(const QString key, const QString title, Qt::PenStyle defaultLineStyle, QwtSymbol::Style defaultMarker) :
    BlackchirpPlotCurveBase(key,title,defaultLineStyle,defaultMarker)
{
}

BCEvenSpacedCurveBase::~BCEvenSpacedCurveBase()
{
}

double BCEvenSpacedCurveBase::xVal(int i) const
{
    return xFirst() + spacing()*static_cast<double>(i);
}

int BCEvenSpacedCurveBase::indexBefore(double xVal) const
{
    return qMin(numPoints(),static_cast<int>((xVal - xFirst()) / spacing() ));
}

QVector<QPointF> BCEvenSpacedCurveBase::_filter(int w, const QwtScaleMap map)
{
    auto d = yData();
    d.detach();
    auto s = d.size();

    if(s < 2.5*w)
    {
        QVector<QPointF> out;
        out.reserve(s);
        for(int i=0; i<s; ++i)
            out.append({xVal(i),d.at(i)});
        return out;
    }

    QVector<QPointF> filtered;
    filtered.reserve(2*w+2);

    int firstPixel = 0;
    int lastPixel = w;
    int inc = 1;

    //XOR operation
    if(!(spacing() < 0.0) != !(map.invTransform(0) > map.invTransform(w)))
    {
        firstPixel = w;
        lastPixel = 0;
        inc = -1;
    }

    auto i = qMax(indexBefore(map.invTransform(firstPixel)),0);

    //curve is out of range of the plot. return empty array
    if(i >= s)
        return filtered;

    //add previous point to output array for smooth edge behavior
    if(i > 0)
        filtered.append({xVal(i-1),d.at(i-1)});

    //at this point, dataIndex is at the first point within the range of the plot. loop over pixels, compressing data
    for(int pixel = firstPixel; pixel!=(lastPixel+inc); pixel+=inc)
    {
        auto min = d.at(i);
        auto max = d.at(i);

        int numPnts = 0;
        int nextPixelIndex = qMin(indexBefore(map.invTransform(pixel+(double)inc))+1,s-1);

        while(i < nextPixelIndex)
        {
            min = qMin(d.at(i),min);
            max = qMax(d.at(i),max);

            ++i;
            ++numPnts;
        }


        if(numPnts == 1)
            filtered.append({xVal(i-1),d.at(i-1)});
        else if (numPnts > 1)
        {
            auto x = map.invTransform(pixel);
            filtered.append({x,min});
            filtered.append({x,max});
        }

    }

    if(i < s)
        filtered.append({xVal(i),d.at(i)});

    return filtered;



}

BlackchirpFTCurve::BlackchirpFTCurve(const QString key, const QString title, Qt::PenStyle defaultLineStyle, QwtSymbol::Style defaultMarker) :
    BCEvenSpacedCurveBase(key,title,defaultLineStyle,defaultMarker), p_mutex(new QMutex)
{

}

BlackchirpFTCurve::~BlackchirpFTCurve()
{
    delete p_mutex;
}

void BlackchirpFTCurve::setCurrentFt(const Ft f)
{
    QMutexLocker l(p_mutex);
    d_currentFt = f;
}


QRectF BlackchirpFTCurve::boundingRect() const
{
    QMutexLocker l(p_mutex);
    if(d_currentFt.isEmpty())
        return QRectF(1.0,1.0,-2.0,-2.0);

    QRectF out;
    out.setLeft(d_currentFt.minFreq());
    out.setRight(d_currentFt.maxFreq());
    out.setTop(d_currentFt.yMin());
    out.setBottom(d_currentFt.yMax());

    return out;
}

QVector<QPointF> BlackchirpFTCurve::curveData() const
{
    QMutexLocker l(p_mutex);
    return d_currentFt.toVector();
}

double BlackchirpFTCurve::xFirst() const
{
    QMutexLocker l(p_mutex);
    return d_currentFt.minFreq();
}

double BlackchirpFTCurve::spacing() const
{
    QMutexLocker l(p_mutex);
    return d_currentFt.xSpacing();
}

int BlackchirpFTCurve::numPoints() const
{
    QMutexLocker l(p_mutex);
    return d_currentFt.size();
}

QVector<double> BlackchirpFTCurve::yData()
{
    QMutexLocker l(p_mutex);
    return d_currentFt.yData();
}

BlackchirpFIDCurve::BlackchirpFIDCurve(const QString key, const QString title, Qt::PenStyle defaultLineStyle, QwtSymbol::Style defaultMarker) :
    BCEvenSpacedCurveBase(key,title,defaultLineStyle,defaultMarker), p_mutex(new QMutex)
{
}

BlackchirpFIDCurve::~BlackchirpFIDCurve()
{
}

void BlackchirpFIDCurve::setCurrentFid(const QVector<double> d, double spacing, double min, double max)
{
    QMutexLocker l(p_mutex);
    d_fidData = d;
    d_spacing = spacing;
    d_min = min;
    d_max = max;
}


QRectF BlackchirpFIDCurve::boundingRect() const
{
    QMutexLocker l(p_mutex);
    if(d_fidData.isEmpty())
        return QRectF(1.0,1.0,-2.0,-2.0);

    QRectF out;
    out.setLeft(0.0);
    out.setRight(d_spacing*d_fidData.size());
    out.setTop(d_min);
    out.setBottom(d_max);

    return out;
}

QVector<QPointF> BlackchirpFIDCurve::curveData() const
{
    QVector<QPointF> out;

    QMutexLocker l(p_mutex);
    auto s = d_fidData.size();
    out.reserve(s);
    for(int i=0; i<s; ++i)
        out.append({d_spacing*i,d_fidData.at(i)});

    return out;
}

double BlackchirpFIDCurve::xFirst() const
{
    return 0.0;
}

double BlackchirpFIDCurve::spacing() const
{
    QMutexLocker l(p_mutex);
    return d_spacing;
}

int BlackchirpFIDCurve::numPoints() const
{
    QMutexLocker l(p_mutex);
    return d_fidData.size();
}

QVector<double> BlackchirpFIDCurve::yData()
{
    QMutexLocker l(p_mutex);
    return d_fidData;
}
