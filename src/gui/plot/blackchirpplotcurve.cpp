#include "blackchirpplotcurve.h"

#include <QPalette>

BlackchirpPlotCurve::BlackchirpPlotCurve(const QString name,Qt::PenStyle defaultLineStyle, QwtSymbol::Style defaultMarker) :
    SettingsStorage({BC::Key::bcCurve,name},General,false), d_min(0.0), d_max(0.0)
{
    setTitle(name);

    getOrSetDefault(BC::Key::bcCurveStyle,static_cast<int>(defaultLineStyle));
    getOrSetDefault(BC::Key::bcCurveMarker,static_cast<int>(defaultMarker));

    configurePen();
    configureSymbol();
    setRenderHint(QwtPlotItem::RenderAntialiased);

    setAxes(get<QwtPlot::Axis>(BC::Key::bcCurveAxisX,QwtPlot::xBottom),
            get<QwtPlot::Axis>(BC::Key::bcCurveAxisY,QwtPlot::yLeft));
    setVisible(get<bool>(BC::Key::bcCurveVisible,true));

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
    d_data = d;
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

void BlackchirpPlotCurve::filter()
{
    auto p = plot();
    int w = p->canvas()->width();

    if(d_data.size() < 2.5*w)
    {
        setSamples(d_data);
        return;
    }

    auto map = p->canvasMap(xAxis());
    double firstPixel = 0.0;
    double lastPixel = w;

    QVector<QPointF> filtered;
    filtered.reserve(2*w+2);

    auto start = d_data.cbegin();
    auto end = d_data.cend();

    int inc = 1;
    if(d_data.first().x() > d_data.last().x())
    {
        qSwap(start,end);
        inc = -1;
    }
    auto it = start;



    //find first data point that is in the range of the plot
//    int dataIndex = 0;
    while(map.transform(it->x()) < firstPixel && it != end)
        it += inc;

//    while(dataIndex+1 < d_data.size() && map.transform(d_data.at(dataIndex).x()) < firstPixel)
//        dataIndex++;

    //add the previous point to the filtered array
    //this will make sure the curve always goes to the edge of the plot
    if(it != start)
    {
        it -= inc;
        filtered.append(it->toPoint());
        it += inc;
    }

//    if(dataIndex-1 >= 0)
//        filtered.append(d_data.at(dataIndex-1));

    if(it != end)
    {
        d_min = it->y();
        d_max = it->y();
    }

//    if(dataIndex < d_data.size())
//    {
//        d_min = d_data.at(dataIndex).y();
//        d_max = d_data.at(dataIndex).y();
//    }

    //at this point, dataIndex is at the first point within the range of the plot. loop over pixels, compressing data
    for(double pixel = firstPixel; pixel<lastPixel; pixel+=1.0)
    {
//        auto min = d_data.at(dataIndex).y();
//        auto max = min;

        auto min = it->y();
        auto max = it->y();

        int numPnts = 0;
        double nextPixelX = map.invTransform(pixel+1.0);
//        while(dataIndex+1 < d_data.size() && d_data.at(dataIndex).x() < nextPixelX)
//        {
//            auto pt = d_data.at(dataIndex);
//            min = qMin(pt.y(),min);
//            max = qMax(pt.y(),max);

//            dataIndex++;
//            numPnts++;
//        }

        while(it != end && it->x() < nextPixelX)
        {
            min = qMin(it->y(),min);
            max = qMax(it->y(),max);

            it += inc;
            numPnts++;
        }

//        if(numPnts == 1)
//            filtered.append(d_data.at(dataIndex-1));
//        else if (numPnts > 1)
//        {
//            QPointF first(map.invTransform(pixel),min);
//            QPointF second(map.invTransform(pixel),max);
//            filtered.append(first);
//            filtered.append(second);
//        }

        if(numPnts == 1)
        {
            it -= inc;
            filtered.append(it->toPoint());
            it += inc;
        }
        else if (numPnts > 1)
        {
            filtered.append({map.invTransform(pixel),min});
            filtered.append({map.invTransform(pixel),max});
        }

        d_min = qMin(d_min,min);
        d_max = qMax(d_max,max);

    }

//    if(dataIndex < d_data.size())
//        filtered.append(d_data.at(dataIndex));

    if(it != end)
        filtered.append(it->toPoint());

    setSamples(filtered);

}
