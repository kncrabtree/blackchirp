#include "blackchirpplotcurve.h"

#include <QPalette>
#include <QMutex>

BlackchirpPlotCurve::BlackchirpPlotCurve(const QString key, const QString title, Qt::PenStyle defaultLineStyle, QwtSymbol::Style defaultMarker) :
    SettingsStorage({BC::Key::bcCurve,key},General), d_key{key},
    p_samplesMutex{new QMutex}, p_dataMutex{new QMutex}
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

BlackchirpPlotCurve::~BlackchirpPlotCurve()
{
    delete p_samplesMutex;
    delete p_dataMutex;
}

void BlackchirpPlotCurve::setColor(const QColor c)
{
    set(BC::Key::bcCurveColor,c);
    configurePen();
    configureSymbol();
}

void BlackchirpPlotCurve::setLineThickness(double t)
{
    set(BC::Key::bcCurveThickness,t);
    configurePen();
}

void BlackchirpPlotCurve::setLineStyle(Qt::PenStyle s)
{
    set(BC::Key::bcCurveStyle,static_cast<int>(s));
    configurePen();
}

void BlackchirpPlotCurve::setMarkerStyle(QwtSymbol::Style s)
{
    set(BC::Key::bcCurveMarker,static_cast<int>(s));
    configureSymbol();
}

void BlackchirpPlotCurve::setMarkerSize(int s)
{
    set(BC::Key::bcCurveMarkerSize,s);
    configureSymbol();
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

void BlackchirpPlotCurve::setCurveVisible(bool v)
{
    set(BC::Key::bcCurveVisible,v);
    setVisible(v);
}

void BlackchirpPlotCurve::setCurveAxisX(QwtPlot::Axis a)
{
    set(BC::Key::bcCurveAxisX,static_cast<int>(a));
    setXAxis(a);
}

void BlackchirpPlotCurve::setCurveAxisY(QwtPlot::Axis a)
{
    set(BC::Key::bcCurveAxisY,static_cast<int>(a));
    setYAxis(a);
}

void BlackchirpPlotCurve::setCurvePlotIndex(int i)
{
    set(BC::Key::bcCurvePlotIndex,i);
}

void BlackchirpPlotCurve::updateFromSettings()
{
    configurePen();
    configureSymbol();
    setAxes(get<QwtPlot::Axis>(BC::Key::bcCurveAxisX,QwtPlot::xBottom),
            get<QwtPlot::Axis>(BC::Key::bcCurveAxisY,QwtPlot::yLeft));
    setVisible(get<bool>(BC::Key::bcCurveVisible,true));
}

void BlackchirpPlotCurve::configurePen()
{
    QPen p;
    QPalette pal;
    p.setColor(get<QColor>(BC::Key::bcCurveColor,pal.color(QPalette::BrightText)));
    p.setWidthF(get<double>(BC::Key::bcCurveThickness,1.0));
    p.setStyle(get<Qt::PenStyle>(BC::Key::bcCurveStyle,Qt::SolidLine));
    setPen(p);
}

void BlackchirpPlotCurve::configureSymbol()
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

void BlackchirpPlotCurve::setSamples(const QVector<QPointF> d)
{
    QMutexLocker l(p_samplesMutex);
    QwtPlotCurve::setSamples(d);

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

void BlackchirpPlotCurve::filter(int w, const QwtScaleMap map)
{
    p_dataMutex->lock();
    auto size = d_curveData.size();
    QVector<QPointF> d = d_curveData;
    d.detach();

//    d.reserve(size);
//    for(auto const &p : d_curveData)
//        d.append({p.x(),p.y()});
    p_dataMutex->unlock();

    if(size < 2.5*w)
    {
        p_dataMutex->unlock();
        d_boundingRect = boundingRect();
        setSamples(d);
        return;
    }

    double firstPixel = 0.0;
    double lastPixel = w;

    QVector<QPointF> filtered;
    filtered.reserve(2*w+2);

    auto start = d.cbegin();
    auto end = d.cend();

    int inc = 1;
    auto firstx = d.first().x();
    auto lastx = d.last().x();
    if(firstx > lastx)
    {
        qSwap(start,end);
        inc = -1;
        start--;
        end--;
    }
    auto it = start;

//    d_boundingRect.setLeft(qMin(firstx,lastx));
//    d_boundingRect.setRight(qMax(firstx,lastx));

    //find first data point that is in the range of the plot
    while(it != end && map.transform(it->x()) < firstPixel)
        it += inc;

    //add the previous point to the filtered array
    //this will make sure the curve always goes to the edge of the plot
    if(it != start)
    {
        it -= inc;
        filtered.append({it->x(),it->y()});
        it += inc;
    }

//    if(it != end)
//    {
//        d_boundingRect.setTop(it->y());
//        d_boundingRect.setBottom(it->y());
//    }

    //at this point, dataIndex is at the first point within the range of the plot. loop over pixels, compressing data
    for(double pixel = firstPixel; pixel<lastPixel; pixel+=1.0)
    {
        auto min = it->y();
        auto max = it->y();

        int numPnts = 0;
        double nextPixelX = map.invTransform(pixel+1.0);

        while(it != end && it->x() < nextPixelX)
        {
            min = qMin(it->y(),min);
            max = qMax(it->y(),max);

            it += inc;
            numPnts++;
        }


        if(numPnts == 1)
        {
            it -= inc;
            filtered.append({it->x(),it->y()});
            it += inc;
        }
        else if (numPnts > 1)
        {
            filtered.append({map.invTransform(pixel),min});
            filtered.append({map.invTransform(pixel),max});
        }

    }

    if(it != end)
        filtered.append({it->x(),it->y()});

    setSamples(filtered);

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


void BlackchirpPlotCurve::draw(QPainter *painter, const QwtScaleMap &xMap, const QwtScaleMap &yMap, const QRectF &canvasRect) const
{
    QMutexLocker l(p_samplesMutex);
    QwtPlotSeriesItem::draw(painter,xMap,yMap,canvasRect);
}
