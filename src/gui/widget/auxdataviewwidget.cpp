#include "auxdataviewwidget.h"

#include <QInputDialog>
#include <QColorDialog>
#include <QMenu>
#include <QActionGroup>
#include <QMouseEvent>
#include <QGridLayout>

#include <qwt6/qwt_date.h>
#include <qwt6/qwt_plot_curve.h>

#include <gui/plot/trackingplot.h>
#include <gui/plot/blackchirpplotcurve.h>
#include <data/datastructs.h>
#include <data/storage/blackchirpcsv.h>


AuxDataViewWidget::AuxDataViewWidget(const QString name, QWidget *parent, bool viewOnly) :
    QWidget(parent), SettingsStorage(viewOnly ? name + BC::Key::viewonly : name,General),
    d_name(viewOnly ? name + BC::Key::viewonly : name), d_viewMode(viewOnly)
{    
    int n = getOrSetDefault(BC::Key::numPlots,4);
    int numPlots = qBound(1,n,9);
    if(numPlots != n)
        set(BC::Key::numPlots,numPlots);

    for(int i=0; i<numPlots;i++)
        addNewPlot();

    configureGrid();
}

AuxDataViewWidget::~AuxDataViewWidget()
{

}

void AuxDataViewWidget::initializeForExperiment()
{
    d_plotCurves.clear();

    for(int i=0; i<d_allPlots.size(); i++)
    {
        d_allPlots[i]->resetPlot();
        d_allPlots[i]->autoScale();
    }
}

void AuxDataViewWidget::pointUpdated(const AuxDataStorage::AuxDataMap m, const QDateTime t)
{
    double x = QwtDate::toDouble(t);

    for(auto &[key,val] : m)
    {
        //first, determine if the QVariant contains a number
        //no need to plot the data if it's not a number
        bool ok = false;
        double value = val.toDouble(&ok);
        if(!ok)
            continue;

        //create point to be plotted
        QPointF newPoint(x,value);

        //locate curve by name and append point
        bool foundCurve = false;

        auto l = key.split(".",QString::SkipEmptyParts);
        QString realKey = key;
        if(l.size() >= 2)
            realKey = QString("%1.%2").arg(l.constFirst(),l.constLast());

        for(auto c : d_plotCurves)
        {
            if(realKey == c->key())
            {
                c->appendPoint(newPoint);

                if(c->isVisible())
                    c->plot()->replot();

                foundCurve = true;
                break;
            }

            purgeOldPoints(c);
        }

        if(foundCurve)
            continue;

        //if we reach this point, a new curve and metadata struct needs to be created
        QString title = realKey;
        if(l.size() > 3)
            title = l.at(l.size()-2);

        BlackchirpPlotCurve *c = new BlackchirpPlotCurve(realKey,title);
        c->appendPoint({x,value});
        c->attach(d_allPlots.at(c->plotIndex() % d_allPlots.size()));
        if(c->isVisible())
            c->plot()->replot();

        d_plotCurves.append(c);
    }
}


void AuxDataViewWidget::moveCurveToPlot(BlackchirpPlotCurve *c, int newPlotIndex)
{
    auto oldPlot = c->plot();
    c->detach();
    c->setCurvePlotIndex(newPlotIndex % d_allPlots.size());
    c->attach(d_allPlots.at(c->plotIndex()));

    oldPlot->replot();
    c->plot()->replot();
}

void AuxDataViewWidget::pushXAxis(int sourcePlotIndex)
{
    if(sourcePlotIndex < 0 || sourcePlotIndex >= d_allPlots.size())
        return;

    const QwtScaleDiv b = d_allPlots.at(sourcePlotIndex)->axisScaleDiv(QwtPlot::xBottom);
    const QwtScaleDiv t = d_allPlots.at(sourcePlotIndex)->axisScaleDiv(QwtPlot::xTop);

    for(int i=0; i<d_allPlots.size(); i++)
    {
        if(i != sourcePlotIndex)
            d_allPlots.at(i)->setXRanges(b,t);
    }
}

void AuxDataViewWidget::autoScaleAll()
{
    for(int i=0; i<d_allPlots.size(); i++)
        d_allPlots.at(i)->autoScale();
}

void AuxDataViewWidget::changeNumPlots(int newNum)
{
    if(newNum == d_allPlots.size())
        return;

    if(!d_viewMode)
        set(BC::Key::numPlots,newNum);

    if(newNum > d_allPlots.size())
    {
        while(d_allPlots.size() < newNum)
            addNewPlot();
    }
    else
    {
        for(auto c: d_plotCurves)
        {
            //reassign any curves that are on graphs about to be removed
            if(c->plotIndex() >= newNum)
                moveCurveToPlot(c,c->plotIndex() % newNum);
        }

        while(newNum < d_allPlots.size())
            delete d_allPlots.takeLast();
    }

    configureGrid();


    for(int i=0; i<d_allPlots.size(); i++)
        d_allPlots.at(i)->replot();

}

void AuxDataViewWidget::addNewPlot()
{
    TrackingPlot *tp = new TrackingPlot(d_name + BC::Key::plot + QString::number(d_allPlots.size()),this);

    tp->setMaxIndex(get<int>(BC::Key::numPlots,1)-1);

//    tp->setMinimumHeight(200);
//    tp->setMinimumWidth(375);
    tp->installEventFilter(this);


    int newPlotIndex = d_allPlots.size();
    connect(tp,&ZoomPanPlot::curveMoveRequested,this,&AuxDataViewWidget::moveCurveToPlot);
    connect(tp,&TrackingPlot::axisPushRequested,this,[=](){ pushXAxis(newPlotIndex); });
    connect(tp,&TrackingPlot::autoScaleAllRequested,this,&AuxDataViewWidget::autoScaleAll);

    d_allPlots.append(tp);

}

void AuxDataViewWidget::configureGrid()
{
    if(d_allPlots.size() < 1)
        return;

    if(p_gridLayout != nullptr)
        delete p_gridLayout;

    p_gridLayout = new QGridLayout();
    setLayout(p_gridLayout);

    switch(d_allPlots.size())
    {
    case 1:
        p_gridLayout->addWidget(d_allPlots[0],0,0,1,1);
        break;
    case 2:
        p_gridLayout->addWidget(d_allPlots[0],0,0,1,1);
        p_gridLayout->addWidget(d_allPlots[1],1,0,1,1);
        p_gridLayout->setRowStretch(0,1);
        p_gridLayout->setRowStretch(1,1);
        break;
    case 3:
        p_gridLayout->addWidget(d_allPlots[0],0,0,1,1);
        p_gridLayout->addWidget(d_allPlots[1],1,0,1,1);
        p_gridLayout->addWidget(d_allPlots[2],2,0,1,1);
        p_gridLayout->setRowStretch(0,1);
        p_gridLayout->setRowStretch(1,1);
        p_gridLayout->setRowStretch(2,1);
        p_gridLayout->setColumnStretch(0,1);
        break;
    case 4:
        p_gridLayout->addWidget(d_allPlots[0],0,0,1,1);
        p_gridLayout->addWidget(d_allPlots[1],0,1,1,1);
        p_gridLayout->addWidget(d_allPlots[2],1,0,1,1);
        p_gridLayout->addWidget(d_allPlots[3],1,1,1,1);
        p_gridLayout->setRowStretch(0,1);
        p_gridLayout->setRowStretch(1,1);
        p_gridLayout->setColumnStretch(0,1);
        p_gridLayout->setColumnStretch(1,1);
        break;
    case 5:
        p_gridLayout->addWidget(d_allPlots[0],0,0,1,2);
        p_gridLayout->addWidget(d_allPlots[1],0,2,1,2);
        p_gridLayout->addWidget(d_allPlots[2],0,4,1,2);
        p_gridLayout->addWidget(d_allPlots[3],1,0,1,3);
        p_gridLayout->addWidget(d_allPlots[4],1,3,1,3);
        p_gridLayout->setRowStretch(0,1);
        p_gridLayout->setRowStretch(1,1);
        for(int i=0; i<6; ++i)
            p_gridLayout->setColumnStretch(i,1);
        break;
    case 6:
        p_gridLayout->addWidget(d_allPlots[0],0,0,1,1);
        p_gridLayout->addWidget(d_allPlots[1],0,1,1,1);
        p_gridLayout->addWidget(d_allPlots[2],0,2,1,1);
        p_gridLayout->addWidget(d_allPlots[3],1,0,1,1);
        p_gridLayout->addWidget(d_allPlots[4],1,1,1,1);
        p_gridLayout->addWidget(d_allPlots[5],1,2,1,1);
        p_gridLayout->setRowStretch(0,1);
        p_gridLayout->setRowStretch(1,1);
        p_gridLayout->setColumnStretch(0,1);
        p_gridLayout->setColumnStretch(1,1);
        p_gridLayout->setColumnStretch(2,1);
        break;
    case 7:
        p_gridLayout->addWidget(d_allPlots[0],0,0,1,2);
        p_gridLayout->addWidget(d_allPlots[1],0,2,1,2);
        p_gridLayout->addWidget(d_allPlots[2],0,4,1,2);
        p_gridLayout->addWidget(d_allPlots[3],1,0,1,3);
        p_gridLayout->addWidget(d_allPlots[4],1,3,1,3);
        p_gridLayout->addWidget(d_allPlots[5],2,0,1,3);
        p_gridLayout->addWidget(d_allPlots[6],2,3,1,3);
        p_gridLayout->setRowStretch(0,1);
        p_gridLayout->setRowStretch(1,1);
        p_gridLayout->setRowStretch(2,1);
        for(int i=0; i<6; ++i)
            p_gridLayout->setColumnStretch(i,1);
        break;
    case 8:
        p_gridLayout->addWidget(d_allPlots[0],0,0,1,1);
        p_gridLayout->addWidget(d_allPlots[1],0,1,1,1);
        p_gridLayout->addWidget(d_allPlots[2],0,2,1,1);
        p_gridLayout->addWidget(d_allPlots[3],0,3,1,1);
        p_gridLayout->addWidget(d_allPlots[4],1,0,1,1);
        p_gridLayout->addWidget(d_allPlots[5],1,1,1,1);
        p_gridLayout->addWidget(d_allPlots[6],1,2,1,1);
        p_gridLayout->addWidget(d_allPlots[7],1,3,1,1);
        p_gridLayout->setRowStretch(0,1);
        p_gridLayout->setRowStretch(1,1);
        p_gridLayout->setColumnStretch(0,1);
        p_gridLayout->setColumnStretch(1,1);
        p_gridLayout->setColumnStretch(2,1);
        p_gridLayout->setColumnStretch(3,1);
        break;
    case 9:
    default:
        p_gridLayout->addWidget(d_allPlots[0],0,0,1,1);
        p_gridLayout->addWidget(d_allPlots[1],0,1,1,1);
        p_gridLayout->addWidget(d_allPlots[2],0,2,1,1);
        p_gridLayout->addWidget(d_allPlots[3],1,0,1,1);
        p_gridLayout->addWidget(d_allPlots[4],1,1,1,1);
        p_gridLayout->addWidget(d_allPlots[5],1,2,1,1);
        p_gridLayout->addWidget(d_allPlots[6],2,0,1,1);
        p_gridLayout->addWidget(d_allPlots[7],2,1,1,1);
        p_gridLayout->addWidget(d_allPlots[8],2,2,1,1);
        p_gridLayout->setRowStretch(0,1);
        p_gridLayout->setRowStretch(1,1);
        p_gridLayout->setRowStretch(2,1);
        p_gridLayout->setColumnStretch(0,1);
        p_gridLayout->setColumnStretch(1,1);
        p_gridLayout->setColumnStretch(2,1);
        break;
    }

}

RollingDataWidget::RollingDataWidget(const QString name, QWidget *parent) : AuxDataViewWidget(name,parent,false)
{
    d_historyDuration = get(BC::Key::history,12);
}

void RollingDataWidget::pointUpdated(const AuxDataStorage::AuxDataMap m, const QDateTime dt)
{
    QDir d = BlackchirpCSV::trackingDir();
    auto year = QString::number(dt.date().year());
    auto month = QString::number(dt.date().month()).rightJustified(2,'0');
    if(!d.cd(year))
    {
        d.mkdir(year);
        d.cd(year);
    }
    if(!d.cd(year+month))
    {
        d.mkdir(year+month);
        d.cd(year+month);
    }

    for(auto &[key,val] : m)
    {
        bool writeheader = false;
        QFile f(d.absoluteFilePath(key)+".csv");
        QTextStream t(&f);
        if(!f.exists())
            writeheader = true;

        if(f.open(QIODevice::Append|QIODevice::Text))
        {
            if(writeheader)
                BlackchirpCSV::writeLine(t,{"timestamp","epochtime",key});
            BlackchirpCSV::writeLine(t,{dt.toString(),dt.toSecsSinceEpoch(),val});
        }
    }

    AuxDataViewWidget::pointUpdated(m,dt);
}

void RollingDataWidget::purgeOldPoints(BlackchirpPlotCurve *c)
{
    auto d = c->curveData();
    int first = 0;
    if(d.constFirst().x() < QwtDate::toDouble(QDateTime::currentDateTime().addSecs(-5400*d_historyDuration)))
    {
        auto cutoff = QwtDate::toDouble(QDateTime::currentDateTime().addSecs(-3600*d_historyDuration));
        while(first < d.size() && d.at(first).x() < cutoff)
            first++;

        d = d.mid(first);
        c->setCurveData(d);
        static_cast<ZoomPanPlot*>(c->plot())->overrideAxisAutoScaleRange(
                    QwtPlot::xBottom,cutoff,QwtDate::toDouble(QDateTime::currentDateTime()));
        static_cast<ZoomPanPlot*>(c->plot())->overrideAxisAutoScaleRange(
                    QwtPlot::xTop,cutoff,QwtDate::toDouble(QDateTime::currentDateTime()));
    }
}

void RollingDataWidget::setHistoryDuration(int d)
{
    d_historyDuration = d;
    set(BC::Key::history,d);
    for(auto c : d_plotCurves)
    {
        purgeOldPoints(c);
        if(c->isVisible())
            c->plot()->replot();
    }
}
