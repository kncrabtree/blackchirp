#include "trackingviewwidget.h"
#include <QSettings>
#include <QInputDialog>
#include <qwt6/qwt_date.h>

TrackingViewWidget::TrackingViewWidget(QWidget *parent) :
    QWidget(parent)
{    
    QSettings s;
    int numPlots = qBound(1,s.value(QString("trackingWidget/numPlots"),4).toInt(),9);
    for(int i=0; i<numPlots;i++)
        addNewPlot();

    d_xRange.first = 0.0;
    d_xRange.second = 0.0;

    configureGrid();
}

TrackingViewWidget::~TrackingViewWidget()
{

}

void TrackingViewWidget::initializeForExperiment()
{
    d_plotCurves.clear();
    d_xRange.first = QwtDate::toDouble(QDateTime::currentDateTime().addSecs(-1));
    d_xRange.second = QwtDate::toDouble(QDateTime::currentDateTime());

    for(int i=0; i<d_allPlots.size(); i++)
    {
        d_allPlots[i]->resetPlot();
        d_allPlots[i]->setAxisAutoScaleRange(QwtPlot::xBottom,d_xRange.first,d_xRange.second);
        d_allPlots[i]->setAxisAutoScaleRange(QwtPlot::xTop,d_xRange.first,d_xRange.second);
        d_allPlots[i]->autoScale();
    }
}

void TrackingViewWidget::pointUpdated(const QList<QPair<QString, QVariant> > list)
{
    double x = QwtDate::toDouble(QDateTime::currentDateTime());
    if(d_plotCurves.isEmpty())
    {
        d_xRange.first = x;
        d_xRange.second = x;

        for(int i=0;i<d_allPlots.size(); i++)
        {
            d_allPlots[i]->setAxisAutoScaleRange(QwtPlot::xBottom,x,x);
            d_allPlots[i]->setAxisAutoScaleRange(QwtPlot::xTop,x,x);
        }
    }
    else
    {
        d_xRange.second = x;

        for(int i=0;i<d_allPlots.size(); i++)
        {
            d_allPlots[i]->setAxisAutoScaleMax(QwtPlot::xBottom,x);
            d_allPlots[i]->setAxisAutoScaleMax(QwtPlot::xTop,x);
        }
    }

    for(int i=0; i<list.size(); i++)
    {
        //first, determine if the QVariant contains a number
        //no need to plot the data if it's not a number
        bool ok = false;
        double value = list.at(i).second.toDouble(&ok);
        if(!ok)
            continue;

        //create point to be plotted
        QPointF newPoint(x,value);

        //locate curve by name and append point
        bool foundCurve = false;
        for(int j=0; j<d_plotCurves.size(); j++)
        {
            if(list.at(i).first == d_plotCurves.at(j).name)
            {
                d_plotCurves[j].data.append(newPoint);
                d_plotCurves[j].min = qMin(d_plotCurves.at(j).min,value);
                d_plotCurves[j].max = qMax(d_plotCurves.at(j).max,value);
                d_plotCurves[j].curve->setSamples(d_plotCurves.at(j).data);
                if(d_plotCurves.at(j).isVisible)
                {
                    d_allPlots.at(d_plotCurves.at(j).plotIndex)->expandAutoScaleRange(
                                d_plotCurves.at(j).axis,d_plotCurves.at(j).min,d_plotCurves.at(j).max);
                    d_allPlots.at(d_plotCurves.at(j).plotIndex)->replot();
                }

                foundCurve = true;
                break;
            }
        }

        if(foundCurve)
            continue;

        //if we reach this point, a new curve and metadata struct need to be created
        QSettings s;
        CurveMetaData md;
        md.data.reserve(100);
        md.data.append(newPoint);

        md.name = list.at(i).first;

        //Create curve
        QwtPlotCurve *c = new QwtPlotCurve(md.name);
        c->setRenderHint(QwtPlotItem::RenderAntialiased);
        md.curve = c;
        c->setSamples(md.data);

        s.beginGroup(QString("trackingWidget/curves/%1").arg(md.name));

        md.axis = s.value(QString("axis"),QwtPlot::yLeft).value<QwtPlot::Axis>();
        md.plotIndex = s.value(QString("plotIndex"),d_plotCurves.size()).toInt() % d_allPlots.size();
        md.isVisible = s.value(QString("isVisible"),true).toBool();

        c->setAxes(QwtPlot::xBottom,md.axis);
        c->setVisible(md.isVisible);

        QColor color = s.value(QString("color"),palette().color(QPalette::Text)).value<QColor>();
        c->setPen(color);
        c->attach(d_allPlots.at(md.plotIndex));
        d_allPlots.at(md.plotIndex)->initializeLabel(md.curve,md.isVisible);

        s.endGroup();

        md.min = value;
        md.max = value;
        if(md.isVisible)
            d_allPlots.at(md.plotIndex)->setAxisAutoScaleRange(md.axis,md.min,md.max);

        d_plotCurves.append(md);
        d_allPlots.at(md.plotIndex)->replot();

    }
}

void TrackingViewWidget::changeNumPlots()
{
    bool ok = true;
    int newNum = QInputDialog::getInt(this,QString("BC: Change Number of Tracking Plots"),QString("Number of plots:"),d_allPlots.size(),1,9,1,&ok);

    if(!ok || newNum == d_allPlots.size())
        return;

    QSettings s;
    s.setValue(QString("trackingWidget/numPlots"),newNum);
    s.sync();

    if(newNum > d_allPlots.size())
    {
        while(d_allPlots.size() < newNum)
            addNewPlot();
    }
    else
    {
        for(int i=0; i < d_plotCurves.size(); i++)
        {
            //reassign any curves that are on graphs about to be removed
            if(d_plotCurves.at(i).plotIndex >= newNum)
            {
                d_plotCurves.at(i).curve->detach();
                int newPlotIndex = d_plotCurves.at(i).plotIndex % newNum;
                d_plotCurves[i].plotIndex = newPlotIndex;
                d_plotCurves.at(i).curve->attach(d_allPlots.at(newPlotIndex));
            }
        }

        while(newNum < d_allPlots.size())
            delete d_allPlots.takeLast();
    }

    configureGrid();

    for(int i=0; i<d_allPlots.size(); i++)
        d_allPlots.at(i)->replot();

}

void TrackingViewWidget::addNewPlot()
{
    TrackingPlot *tp = new TrackingPlot(this);

    tp->setAxisAutoScaleRange(QwtPlot::xBottom,d_xRange.first,d_xRange.second);
    tp->setAxisAutoScaleRange(QwtPlot::xTop,d_xRange.first,d_xRange.second);

    tp->setMinimumHeight(200);
    tp->setMinimumWidth(375);
    tp->installEventFilter(this);

    //signal-slot connections go here

    d_allPlots.append(tp);

}

void TrackingViewWidget::configureGrid()
{
    if(d_allPlots.size() < 1)
        return;

    if(d_gridLayout != nullptr)
        delete d_gridLayout;

    d_gridLayout = new QGridLayout();
    setLayout(d_gridLayout);

    switch(d_allPlots.size())
    {
    case 1:
        d_gridLayout->addWidget(d_allPlots[0],0,0,1,1);
        break;
    case 2:
        d_gridLayout->addWidget(d_allPlots[0],0,0,1,1);
        d_gridLayout->addWidget(d_allPlots[1],1,0,1,1);
        d_gridLayout->setRowStretch(0,1);
        d_gridLayout->setRowStretch(1,1);
        break;
    case 3:
        d_gridLayout->addWidget(d_allPlots[0],0,0,1,1);
        d_gridLayout->addWidget(d_allPlots[1],0,1,1,1);
        d_gridLayout->addWidget(d_allPlots[2],1,0,1,2);
        d_gridLayout->setRowStretch(0,1);
        d_gridLayout->setRowStretch(1,1);
        d_gridLayout->setColumnStretch(0,1);
        d_gridLayout->setColumnStretch(1,1);
        break;
    case 4:
        d_gridLayout->addWidget(d_allPlots[0],0,0,1,1);
        d_gridLayout->addWidget(d_allPlots[1],0,1,1,1);
        d_gridLayout->addWidget(d_allPlots[2],1,0,1,1);
        d_gridLayout->addWidget(d_allPlots[3],1,1,1,1);
        d_gridLayout->setRowStretch(0,1);
        d_gridLayout->setRowStretch(1,1);
        d_gridLayout->setColumnStretch(0,1);
        d_gridLayout->setColumnStretch(1,1);
        break;
    case 5:
        d_gridLayout->addWidget(d_allPlots[0],0,0,1,1);
        d_gridLayout->addWidget(d_allPlots[1],0,1,1,1);
        d_gridLayout->addWidget(d_allPlots[2],0,2,1,1);
        d_gridLayout->addWidget(d_allPlots[3],1,0,1,1);
        d_gridLayout->addWidget(d_allPlots[4],1,1,1,2);
        d_gridLayout->setRowStretch(0,1);
        d_gridLayout->setRowStretch(1,1);
        d_gridLayout->setColumnStretch(0,1);
        d_gridLayout->setColumnStretch(1,1);
        d_gridLayout->setColumnStretch(2,1);
        break;
    case 6:
        d_gridLayout->addWidget(d_allPlots[0],0,0,1,1);
        d_gridLayout->addWidget(d_allPlots[1],0,1,1,1);
        d_gridLayout->addWidget(d_allPlots[2],0,2,1,1);
        d_gridLayout->addWidget(d_allPlots[3],1,0,1,1);
        d_gridLayout->addWidget(d_allPlots[4],1,1,1,1);
        d_gridLayout->addWidget(d_allPlots[5],1,2,1,1);
        d_gridLayout->setRowStretch(0,1);
        d_gridLayout->setRowStretch(1,1);
        d_gridLayout->setColumnStretch(0,1);
        d_gridLayout->setColumnStretch(1,1);
        d_gridLayout->setColumnStretch(2,1);
        break;
    case 7:
        d_gridLayout->addWidget(d_allPlots[0],0,0,1,1);
        d_gridLayout->addWidget(d_allPlots[1],0,1,1,1);
        d_gridLayout->addWidget(d_allPlots[2],0,2,1,1);
        d_gridLayout->addWidget(d_allPlots[3],0,3,1,1);
        d_gridLayout->addWidget(d_allPlots[4],1,0,1,1);
        d_gridLayout->addWidget(d_allPlots[5],1,1,1,1);
        d_gridLayout->addWidget(d_allPlots[6],1,2,1,2);
        d_gridLayout->setRowStretch(0,1);
        d_gridLayout->setRowStretch(1,1);
        d_gridLayout->setColumnStretch(0,1);
        d_gridLayout->setColumnStretch(1,1);
        d_gridLayout->setColumnStretch(2,1);
        d_gridLayout->setColumnStretch(3,1);
        break;
    case 8:
        d_gridLayout->addWidget(d_allPlots[0],0,0,1,1);
        d_gridLayout->addWidget(d_allPlots[1],0,1,1,1);
        d_gridLayout->addWidget(d_allPlots[2],0,2,1,1);
        d_gridLayout->addWidget(d_allPlots[3],0,3,1,1);
        d_gridLayout->addWidget(d_allPlots[4],1,0,1,1);
        d_gridLayout->addWidget(d_allPlots[5],1,1,1,1);
        d_gridLayout->addWidget(d_allPlots[6],1,2,1,1);
        d_gridLayout->addWidget(d_allPlots[7],1,3,1,1);
        d_gridLayout->setRowStretch(0,1);
        d_gridLayout->setRowStretch(1,1);
        d_gridLayout->setColumnStretch(0,1);
        d_gridLayout->setColumnStretch(1,1);
        d_gridLayout->setColumnStretch(2,1);
        d_gridLayout->setColumnStretch(3,1);
        break;
    case 9:
    default:
        d_gridLayout->addWidget(d_allPlots[0],0,0,1,1);
        d_gridLayout->addWidget(d_allPlots[1],0,1,1,1);
        d_gridLayout->addWidget(d_allPlots[2],0,2,1,1);
        d_gridLayout->addWidget(d_allPlots[3],1,0,1,1);
        d_gridLayout->addWidget(d_allPlots[4],1,1,1,1);
        d_gridLayout->addWidget(d_allPlots[5],1,2,1,1);
        d_gridLayout->addWidget(d_allPlots[6],2,0,1,1);
        d_gridLayout->addWidget(d_allPlots[7],2,1,1,1);
        d_gridLayout->addWidget(d_allPlots[8],2,2,1,1);
        d_gridLayout->setRowStretch(0,1);
        d_gridLayout->setRowStretch(1,1);
        d_gridLayout->setRowStretch(2,1);
        d_gridLayout->setColumnStretch(0,1);
        d_gridLayout->setColumnStretch(1,1);
        d_gridLayout->setColumnStretch(2,1);
        break;
    }

}

void TrackingViewWidget::setAutoScaleYRanges(int plotIndex, QwtPlot::Axis axis)
{
    double min;
    double max;
    bool foundCurve = false;
    for(int i=0;i<d_plotCurves.size(); i++)
    {
        const CurveMetaData c = d_plotCurves.at(i);
        if(c.plotIndex == plotIndex && axis == c.axis && c.isVisible)
        {
            if(!foundCurve)
            {
                foundCurve = true;
                min = c.min;
                max = c.max;
            }
            else
            {
                min = qMin(c.min,min);
                max = qMax(c.max,max);
            }
        }
    }

    if(foundCurve)
        d_allPlots[plotIndex]->setAxisAutoScaleRange(axis,min,max);
    else
        d_allPlots[plotIndex]->setAxisAutoScaleRange(axis,0.0,1.0);
}
