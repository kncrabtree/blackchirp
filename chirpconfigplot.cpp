#include "chirpconfigplot.h"

#include <QSettings>
#include <QColorDialog>
#include <QMouseEvent>
#include <QMenu>

#include <qwt6/qwt_legend.h>
#include <qwt6/qwt_plot_curve.h>

#include "chirpconfig.h"

ChirpConfigPlot::ChirpConfigPlot(QWidget *parent) : ZoomPanPlot(QString("ChirpConfigPlot"),parent), d_protectionEnabled(true), d_ampEnablePulseEnabled(true)
{

    //make axis label font smaller
    this->setAxisFont(QwtPlot::xBottom,QFont(QString("sans-serif"),8));
    this->setAxisFont(QwtPlot::yLeft,QFont(QString("sans-serif"),8));

    //build axis titles with small font. The <html> etc. tags are needed to display the mu character
    QwtText blabel(QString("<html><body>Time (&mu;s)</body></html>"));
    blabel.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::xBottom,blabel);

    QwtText llabel(QString("Chirp (Normalized)"));
    llabel.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::yLeft,llabel);

    QSettings s;
    QPalette pal;

    p_chirpCurve = new QwtPlotCurve(QString("Chirp"));
    QColor color = s.value(QString("chirpColor"),pal.brightText().color()).value<QColor>();
    p_chirpCurve->setPen(QPen(color));
    p_chirpCurve->attach(this);

    p_ampEnableCurve= new QwtPlotCurve(QString("Amp Enable"));
    color = s.value(QString("ampEnableColor"),pal.brightText().color()).value<QColor>();
    p_ampEnableCurve->setPen(QPen(color));
//    p_ampEnableCurve->attach(this);

    p_protectionCurve = new QwtPlotCurve(QString("Protection"));
    color = s.value(QString("protectionColor"),pal.brightText().color()).value<QColor>();
    p_protectionCurve->setPen(QPen(color));
//    p_protectionCurve->attach(this);

    setAxisAutoScaleRange(QwtPlot::yLeft,-1.0,1.0);

    insertLegend(new QwtLegend());

    connect(this,&ChirpConfigPlot::plotRightClicked,this,&ChirpConfigPlot::buildContextMenu);
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
    {
        setAxisAutoScaleRange(QwtPlot::xBottom,0.0,1.0);
        d_chirpData.clear();
        p_ampEnableCurve->setSamples(QVector<QPointF>());
        p_protectionCurve->setSamples(QVector<QPointF>());
        autoScale();
        return;
    }
    else
        setAxisAutoScaleRange(QwtPlot::xBottom,d_chirpData.at(0).x(),d_chirpData.at(d_chirpData.size()-1).x());

    if(as)
        autoScale();

    QVector<QPointF> ampData, protectionData;

    for(int i=0; i<cc.numChirps(); i++)
    {
        double segmentStartTime = cc.chirpInterval()*static_cast<double>(i);
        double twtEnableTime = segmentStartTime + cc.preChirpProtection();
        double chirpEndTime = twtEnableTime + cc.preChirpDelay() + cc.chirpDuration(i);
        double twtEndTime = chirpEndTime + cc.postChirpDelay();
        double protectionEndTime = chirpEndTime + cc.postChirpProtection();

        //build protection data
        if(d_protectionEnabled)
        {
            protectionData.append(QPointF(segmentStartTime,0.0));
            protectionData.append(QPointF(segmentStartTime,1.0));
            protectionData.append(QPointF(protectionEndTime,1.0));
            protectionData.append(QPointF(protectionEndTime,0.0));
        }


        //build Enable data
        if(d_ampEnablePulseEnabled)
        {
            ampData.append(QPointF(segmentStartTime,0.0));
            ampData.append(QPointF(twtEnableTime,0.0));
            ampData.append(QPointF(twtEnableTime,1.0));
            ampData.append(QPointF(twtEndTime,1.0));
            ampData.append(QPointF(twtEndTime,0.0));
            ampData.append(QPointF(protectionEndTime,0.0));
        }

    }

    p_ampEnableCurve->setSamples(ampData);
    p_protectionCurve->setSamples(protectionData);

    filterData();
    replot();
}

void ChirpConfigPlot::buildContextMenu(QMouseEvent *me)
{
    QMenu *menu = contextMenu();

    QAction *chirpAction = menu->addAction(QString("Change chirp color..."));
    connect(chirpAction,&QAction::triggered,[=](){ setCurveColor(p_chirpCurve); });
    if(d_chirpData.isEmpty())
        chirpAction->setEnabled(false);

    QAction *ampAction = menu->addAction(QString("Change amp enable color..."));
    connect(ampAction,&QAction::triggered,[=](){ setCurveColor(p_ampEnableCurve); });
    if(d_chirpData.isEmpty())
        ampAction->setEnabled(false);

    QAction *protAction = menu->addAction(QString("Change protection color..."));
    connect(protAction,&QAction::triggered,[=](){ setCurveColor(p_protectionCurve); });
    if(d_chirpData.isEmpty())
        protAction->setEnabled(false);

    menu->popup(me->globalPos());
}

void ChirpConfigPlot::setProtectionEnabled(bool en)
{
    d_protectionEnabled = en;
    p_protectionCurve->detach();
    if(en)
        p_protectionCurve->attach(this);
}

void ChirpConfigPlot::setAmpEnablePulseEnabled(bool en)
{
    d_ampEnablePulseEnabled = en;
    p_ampEnableCurve->detach();
    if(en)
        p_ampEnableCurve->attach(this);
}

void ChirpConfigPlot::setCurveColor(QwtPlotCurve *c)
{
    if(c != nullptr)
    {
        QString key;
        if(c == p_chirpCurve)
            key = QString("chirpColor");
        if(c == p_ampEnableCurve)
            key = QString("ampEnableColor");
        if(c == p_protectionCurve)
            key = QString("protectionColor");

        if(key.isEmpty())
            return;

        QColor color = QColorDialog::getColor(c->pen().color(),this,QString("Choose color"));
        if(!color.isValid())
            return;

        c->setPen(color);

        QSettings s;
        s.setValue(key,color);
        s.sync();

        replot();
    }
}

void ChirpConfigPlot::filterData()
{
    if(d_chirpData.size() < 2)
    {
        p_chirpCurve->setSamples(d_chirpData);
        return;
    }

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
