#include "fidplot.h"
#include <math.h>

#include <QPalette>
#include <QSettings>
#include <QApplication>
#include <QMenu>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QWidgetAction>
#include <QFormLayout>
#include <QColorDialog>
#include <QMouseEvent>
#include <QFileDialog>
#include <QMessageBox>

#include <qwt6/qwt_plot_canvas.h>
#include <qwt6/qwt_plot_marker.h>
#include <qwt6/qwt_plot_curve.h>

FidPlot::FidPlot(QString id, QWidget *parent) :
    ZoomPanPlot(QString("FidPlot"+id),parent)
{
    //make axis label font smaller
    this->setAxisFont(QwtPlot::xBottom,QFont(QString("sans-serif"),8));
    this->setAxisFont(QwtPlot::yLeft,QFont(QString("sans-serif"),8));

    //build axis titles with small font. The <html> etc. tags are needed to display the mu character
    QwtText blabel(QString("<html><body>Time (&mu;s)</body></html>"));
    blabel.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::xBottom,blabel);

    QwtText llabel(QString("FID "+id));
    llabel.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::yLeft,llabel);

//    this->setAxisScaleDraw(QwtPlot::yLeft,new SciNotationScaleDraw());


    p_curve = new QwtPlotCurve();
    QSettings s;
    s.beginGroup(d_name);
    QColor c = s.value(QString("fidcolor"),palette().color(QPalette::Text)).value<QColor>();
    s.endGroup();
    p_curve->setPen(QPen(c));
    p_curve->setRenderHint(QwtPlotItem::RenderAntialiased);
    p_curve->attach(this);
    p_curve->setVisible(false);

    QwtPlotMarker *chirpStartMarker = new QwtPlotMarker();
    chirpStartMarker->setLineStyle(QwtPlotMarker::VLine);
    chirpStartMarker->setLinePen(QPen(QPalette().color(QPalette::Text)));
    QwtText csLabel(QString("Chirp Start"));
    csLabel.setFont(QFont(QString("sans serif"),6));
    chirpStartMarker->setLabel(csLabel);
    chirpStartMarker->setLabelOrientation(Qt::Vertical);
    chirpStartMarker->setLabelAlignment(Qt::AlignBottom|Qt::AlignRight);
    d_chirpMarkers.first = chirpStartMarker;
    chirpStartMarker->attach(this);
    chirpStartMarker->setVisible(false);

    QwtPlotMarker *chirpEndMarker = new QwtPlotMarker();
    chirpEndMarker->setLineStyle(QwtPlotMarker::VLine);
    chirpEndMarker->setLinePen(QPen(QPalette().color(QPalette::Text)));
    QwtText ceLabel(QString("Chirp End"));
    ceLabel.setFont(QFont(QString("sans serif"),6));
    chirpEndMarker->setLabel(ceLabel);
    chirpEndMarker->setLabelOrientation(Qt::Vertical);
    chirpEndMarker->setLabelAlignment(Qt::AlignTop|Qt::AlignRight);
    d_chirpMarkers.second = chirpEndMarker;
    chirpEndMarker->attach(this);
    chirpEndMarker->setVisible(false);


    QwtPlotMarker *ftStartMarker = new QwtPlotMarker();
    ftStartMarker->setLineStyle(QwtPlotMarker::VLine);
    ftStartMarker->setLinePen(QPen(QPalette().color(QPalette::Text)));
    QwtText ftsLabel(QString(" FT Start "));
    ftsLabel.setFont(QFont(QString("sans serif"),6));
    ftsLabel.setBackgroundBrush(QPalette().window());
    ftsLabel.setColor(QPalette().text().color());
    ftStartMarker->setLabel(ftsLabel);
    ftStartMarker->setLabelOrientation(Qt::Vertical);
    ftStartMarker->setLabelAlignment(Qt::AlignBottom|Qt::AlignRight);
    ftStartMarker->setXValue(0.0);
    ftStartMarker->attach(this);
    ftStartMarker->setVisible(false);
    d_ftMarkers.first = ftStartMarker;

    QwtPlotMarker *ftEndMarker = new QwtPlotMarker();
    ftEndMarker->setLineStyle(QwtPlotMarker::VLine);
    ftEndMarker->setLinePen(QPen(QPalette().color(QPalette::Text)));
    QwtText fteLabel(QString(" FT End "));
    fteLabel.setFont(QFont(QString("sans serif"),6));
    fteLabel.setBackgroundBrush(QPalette().window());
    fteLabel.setColor(QPalette().text().color());
    ftEndMarker->setLabel(fteLabel);
    ftEndMarker->setLabelOrientation(Qt::Vertical);
    ftEndMarker->setLabelAlignment(Qt::AlignTop|Qt::AlignLeft);
    ftEndMarker->setXValue(0.0);
    ftEndMarker->attach(this);
    ftEndMarker->setVisible(false);
    d_ftMarkers.second = ftEndMarker;

    connect(this,&FidPlot::plotRightClicked,this,&FidPlot::buildContextMenu);

    setAxisAutoScaleRange(QwtPlot::yLeft,0.0,1.0);

}

void FidPlot::receiveProcessedFid(const QVector<QPointF> d)
{
    if(d.size() < 2)
        return;

    d_currentFid = d;

    filterData();
    replot();
}

void FidPlot::filterData()
{
    if(d_currentFid.size()<2)
        return;

    double firstPixel = 0.0;
    double lastPixel = canvas()->width();
    QwtScaleMap map = canvasMap(QwtPlot::xBottom);

    QVector<QPointF> filtered;

    //find first data point that is in the range of the plot
    int dataIndex = 0;
    while(dataIndex+1 < d_currentFid.size() && map.transform(d_currentFid.at(dataIndex).x()*1e6) < firstPixel)
        dataIndex++;

    //add the previous point to the filtered array
    //this will make sure the curve always goes to the edge of the plot
    if(dataIndex-1 >= 0)
        filtered.append(d_currentFid.at(dataIndex-1));

    //at this point, dataIndex is at the first point within the range of the plot. loop over pixels, compressing data
    double yMin = d_currentFid.at(dataIndex).y(), yMax = yMin;
    for(double pixel = firstPixel; pixel<lastPixel; pixel+=1.0)
    {
        double min = d_currentFid.at(dataIndex).y(), max = min;
        int minIndex = dataIndex, maxIndex = dataIndex;
        int numPnts = 0;
        double nextPixelX = map.invTransform(pixel+1.0)/1e6;
        while(dataIndex+1 < d_currentFid.size() && d_currentFid.at(dataIndex).x() < nextPixelX)
        {
            if(d_currentFid.at(dataIndex).y() < min)
            {
                min = d_currentFid.at(dataIndex).y();
                minIndex = dataIndex;
            }
            if(d_currentFid.at(dataIndex).y() > max)
            {
                max = d_currentFid.at(dataIndex).y();
                maxIndex = dataIndex;
            }
            dataIndex++;
            numPnts++;
        }
        if(filtered.isEmpty())
        {
            yMin = min;
            yMax = max;
        }
        else
        {
            yMin = qMin(min,yMin);
            yMax = qMax(max,yMax);
        }
        if(numPnts == 1)
            filtered.append(QPointF(d_currentFid.at(dataIndex-1).x()*1e6,d_currentFid.at(dataIndex-1).y()));
        else if (numPnts > 1)
        {
            QPointF first(map.invTransform(pixel),d_currentFid.at(minIndex).y());
            QPointF second(map.invTransform(pixel),d_currentFid.at(maxIndex).y());
            filtered.append(first);
            filtered.append(second);
        }
    }

    if(dataIndex < d_currentFid.size())
    {
        QPointF p = d_currentFid.at(dataIndex);
        p.setX(p.x()*1e6);
        filtered.append(p);
    }

    setAxisAutoScaleRange(QwtPlot::yLeft,yMin,yMax);
    //assign data to curve object
    p_curve->setSamples(filtered);
}

void FidPlot::prepareForExperiment(const Experiment e)
{     
    FtmwConfig c = e.ftmwConfig();
    d_currentFid = QVector<QPointF>();
    p_curve->setSamples(QVector<QPointF>());

    if(!c.isEnabled())
    {
        p_curve->setVisible(false);

        d_chirpMarkers.first->setVisible(false);
        d_chirpMarkers.second->setVisible(false);
        d_ftMarkers.first->setVisible(false);
        d_ftMarkers.second->setVisible(false);

        setAxisAutoScaleRange(QwtPlot::xBottom,0.0,1.0);
        setAxisAutoScaleRange(QwtPlot::yLeft,0.0,1.0);
    }
    else
    {
        p_curve->setVisible(true);

        d_ftMarkers.first->setVisible(true);
        d_ftMarkers.second->setVisible(true);

        double maxTime = c.scopeConfig().recordLength/c.scopeConfig().sampleRate*1e6;
        double ftEnd = d_ftMarkers.second->xValue();
        if(ftEnd < 0.0 || ftEnd < d_ftMarkers.first->xValue() || ftEnd > maxTime)
            d_ftMarkers.second->setXValue(maxTime);

        emit ftStartChanged(d_ftMarkers.first->xValue());
        emit ftEndChanged(d_ftMarkers.second->xValue());

        setAxisAutoScaleRange(QwtPlot::xBottom,0.0,maxTime);
        setAxisAutoScaleRange(QwtPlot::yLeft,0.0,0.0);

        bool displayMarkers = c.isPhaseCorrectionEnabled() || c.isChirpScoringEnabled();
        if(displayMarkers)
        {
            ///TODO: Update this calculation!
            double chirpStart = c.chirpConfig().preChirpGateDelay() + c.chirpConfig().preChirpProtectionDelay() - c.scopeConfig().trigDelay*1e6;
            double chirpEnd = chirpStart + c.chirpConfig().chirpDuration(0);

            d_chirpMarkers.first->setValue(chirpStart,0.0);
            d_chirpMarkers.second->setValue(chirpEnd,0.0);
        }

        d_chirpMarkers.first->setVisible(displayMarkers);
        d_chirpMarkers.second->setVisible(displayMarkers);
    }

    autoScale();
}

void FidPlot::setFtStart(double start)
{
    double v = start;
    if(!d_currentFid.isEmpty())
        v = qBound(0.0,start,qMin(d_ftMarkers.second->value().x(),d_currentFid.last().x()*1e6));

    d_ftMarkers.first->setValue(v,0.0);
    emit ftStartChanged(v);

    QwtPlot::replot();
}

void FidPlot::setFtEnd(double end)
{
    double v = end;
    if(!d_currentFid.isEmpty())
        v = qBound(d_ftMarkers.first->value().x(),end,d_currentFid.last().x()*1e6);

    d_ftMarkers.second->setValue(v,0.0);
    emit ftEndChanged(v);

    QwtPlot::replot();
}

void FidPlot::buildContextMenu(QMouseEvent *me)
{
    if(d_currentFid.size()<2 || !isEnabled())
        return;

    QMenu *menu = contextMenu();

    QAction *colorAct = menu->addAction(QString("Change FID color..."));
    connect(colorAct,&QAction::triggered,this,&FidPlot::changeFidColor);

    menu->popup(me->globalPos());

}

void FidPlot::changeFidColor()
{
    QColor c = QColorDialog::getColor(p_curve->pen().color(),this,QString("Select Color"));
    if(c.isValid())
    {
        p_curve->setPen(c);

        QSettings s;
        s.setValue(QString("fidcolor"),c);
        s.sync();

        replot();
    }
}
