#include "trackingviewwidget.h"

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


TrackingViewWidget::TrackingViewWidget(const QString name, QWidget *parent, bool viewOnly) :
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

TrackingViewWidget::~TrackingViewWidget()
{

}

void TrackingViewWidget::initializeForExperiment()
{
    d_plotCurves.clear();

    for(int i=0; i<d_allPlots.size(); i++)
    {
        d_allPlots[i]->resetPlot();
        d_allPlots[i]->autoScale();
    }
}

void TrackingViewWidget::pointUpdated(const AuxDataStorage::AuxDataMap m, const QDateTime t)
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
        for(auto c : d_plotCurves)
        {
            if(key == c->title().text())
            {
                c->appendPoint(newPoint);

                if(c->isVisible())
                    c->plot()->replot();

                foundCurve = true;
                break;
            }
        }

        if(foundCurve)
            continue;

        //if we reach this point, a new curve and metadata struct needs to be created
        QString realName = BlackChirp::channelNameLookup(key);
        if(realName.isEmpty())
            realName = key;

        BlackchirpPlotCurve *c = new BlackchirpPlotCurve(realName);
        c->appendPoint({x,value});
        c->attach(d_allPlots.at(c->plotIndex() % d_allPlots.size()));
        if(c->isVisible())
            c->plot()->replot();

        d_plotCurves.append(c);
    }
}


void TrackingViewWidget::moveCurveToPlot(BlackchirpPlotCurve* c, int newPlotIndex)
{
    auto oldPlot = c->plot();
    c->detach();
    c->setCurvePlotIndex(newPlotIndex % d_allPlots.size());
    c->attach(d_allPlots.at(c->plotIndex()));

    oldPlot->replot();
    c->plot()->replot();
}

void TrackingViewWidget::pushXAxis(int sourcePlotIndex)
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

void TrackingViewWidget::autoScaleAll()
{
    for(int i=0; i<d_allPlots.size(); i++)
        d_allPlots.at(i)->autoScale();
}

void TrackingViewWidget::changeNumPlots()
{
    bool ok = true;
    int newNum = QInputDialog::getInt(this,QString("BC: Change Number of Tracking Plots"),QString("Number of plots:"),d_allPlots.size(),1,9,1,&ok);

    if(!ok || newNum == d_allPlots.size())
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

void TrackingViewWidget::addNewPlot()
{
    QString name = QString("TrackingPlot%1").arg(d_allPlots.size());
    if(d_viewMode)
        name = QString("TrackingPlotView%1").arg(d_allPlots.size());

    TrackingPlot *tp = new TrackingPlot(d_name + BC::Key::plot + QString::number(d_allPlots.size()),this);

    tp->setMaxIndex(get<int>(BC::Key::numPlots,1)-1);

//    tp->setMinimumHeight(200);
//    tp->setMinimumWidth(375);
    tp->installEventFilter(this);


    int newPlotIndex = d_allPlots.size();
    connect(tp,&ZoomPanPlot::curveMoveRequested,this,&TrackingViewWidget::moveCurveToPlot);
    connect(tp,&TrackingPlot::axisPushRequested,this,[=](){ pushXAxis(newPlotIndex); });
    connect(tp,&TrackingPlot::autoScaleAllRequested,this,&TrackingViewWidget::autoScaleAll);

    d_allPlots.append(tp);

}

void TrackingViewWidget::configureGrid()
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
