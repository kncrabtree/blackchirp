#include "chirpconfigplot.h"
#include <QSettings>

ChirpConfigPlot::ChirpConfigPlot(QWidget *parent) : ZoomPanPlot(QString("ChirpConfigPlot"),parent)
{
    QSettings s;
    QPalette pal;

    p_chirpCurve = new QwtPlotCurve(QString("Chirp"));
    QColor color = s.value(QString("chirpColor"),pal.brightText().color()).value<QColor>();
    p_chirpCurve->setPen(QPen(color));
    p_chirpCurve->attach(this);

    p_twtEnableCurve= new QwtPlotCurve(QString("TWT Enable"));
    color = s.value(QString("twtEnableColor"),pal.brightText().color()).value<QColor>();
    p_twtEnableCurve->setPen(QPen(color));
//    p_twtEnableCurve->attach(this);

    p_protectionCurve = new QwtPlotCurve(QString("Protection"));
    color = s.value(QString("protectionColor"),pal.brightText().color()).value<QColor>();
    p_protectionCurve->setPen(QPen(color));
//    p_protectionCurve->attach(this);

    setAxisAutoScaleRange(QwtPlot::yLeft,-1.0,1.0);
}

ChirpConfigPlot::~ChirpConfigPlot()
{
    detachItems();
}

void ChirpConfigPlot::newChirp(const ChirpConfig cc)
{
    bool as = d_chirpData.isEmpty();

    d_chirpData = cc.getChirpMicroseconds();
    if(d_chirpData.isEmpty())
        setAxisAutoScaleRange(QwtPlot::xBottom,0.0,1.0);
    else
        setAxisAutoScaleRange(QwtPlot::xBottom,d_chirpData.at(0).x(),d_chirpData.at(d_chirpData.size()-1).x());

    if(as)
        autoScale();

    filterData();
    replot();
}

void ChirpConfigPlot::filterData()
{
    if(d_chirpData.size() < 2)
        return;

    double firstPixel = 0.0;
    double lastPixel = canvas()->width();
    QwtScaleMap map = canvasMap(QwtPlot::xBottom);

    QVector<QPointF> filtered;

    //find first data point that is in the range of the plot
    int dataIndex = 0;
    while(dataIndex+1 < d_chirpData.size() && map.transform(d_chirpData.at(dataIndex).x()) < firstPixel)
        dataIndex++;

    //add the previous point to the filtered array
    //this will make sure the curve always goes to the edge of the plot
    if(dataIndex-1 >= 0)
        filtered.append(d_chirpData.at(dataIndex-1));

    //at this point, dataIndex is at the first point within the range of the plot. loop over pixels, compressing data
    for(double pixel = firstPixel; pixel<lastPixel; pixel+=1.0)
    {
        double min = d_chirpData.at(dataIndex).y(), max = min;
        int minIndex = dataIndex, maxIndex = dataIndex;
        int numPnts = 0;
        while(dataIndex+1 < d_chirpData.size() && map.transform(d_chirpData.at(dataIndex).x()) < pixel+1.0)
        {
            if(d_chirpData.at(dataIndex).y() < min)
            {
                min = d_chirpData.at(dataIndex).y();
                minIndex = dataIndex;
            }
            if(d_chirpData.at(dataIndex).y() > max)
            {
                max = d_chirpData.at(dataIndex).y();
                maxIndex = dataIndex;
            }
            dataIndex++;
            numPnts++;
        }
        if(numPnts == 1)
            filtered.append(QPointF(d_chirpData.at(dataIndex-1).x(),d_chirpData.at(dataIndex-1).y()));
        else if (numPnts > 1)
        {
            QPointF first(map.invTransform(pixel),d_chirpData.at(minIndex).y());
            QPointF second(map.invTransform(pixel),d_chirpData.at(maxIndex).y());
            filtered.append(first);
            filtered.append(second);
        }
    }

    if(dataIndex < d_chirpData.size())
    {
        QPointF p = d_chirpData.at(dataIndex);
        p.setX(p.x());
        filtered.append(p);
    }

    //assign data to curve object
    p_chirpCurve->setSamples(filtered);
}
