#include "trackingviewwidget.h"

#include <QSettings>
#include <QInputDialog>
#include <QColorDialog>
#include <QMenu>
#include <QActionGroup>
#include <QMouseEvent>
#include <QGridLayout>
#include <QApplication>

#include <qwt6/qwt_date.h>
#include <qwt6/qwt_plot_curve.h>

#include <src/gui/plot/trackingplot.h>
#include <src/gui/plot/blackchirpplotcurve.h>
#include <src/data/datastructs.h>


TrackingViewWidget::TrackingViewWidget(const QString name, QWidget *parent, bool viewOnly) :
    QWidget(parent), SettingsStorage(viewOnly ? name + BC::Key::viewonly : name,General,false),
    d_name(viewOnly ? name + BC::Key::viewonly : name), d_viewMode(viewOnly)
{    
    int n = get<int>(BC::Key::numPlots,4);
    int numPlots = qBound(1,n,9);
    if(numPlots != n)
        set(BC::Key::numPlots,numPlots);

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
        d_allPlots[i]->autoScale();
    }
}

void TrackingViewWidget::pointUpdated(const QList<QPair<QString, QVariant> > list, bool plot, QDateTime t)
{
    if(!plot)
        return;

    double x = QwtDate::toDouble(t);
    if(d_plotCurves.isEmpty())
    {
        d_xRange.first = x;
        d_xRange.second = x;
    }
    else
    {
        d_xRange.second = x;
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
        QString realName = BlackChirp::channelNameLookup(md.name);
        if(realName.isEmpty())
            realName = md.name;

        BlackchirpPlotCurve *c = new BlackchirpPlotCurve(realName);
        c->setRenderHint(QwtPlotItem::RenderAntialiased);
        md.curve = c;
        c->setCurveData(md.data);

        s.beginGroup(QString("trackingWidget/curves/%1").arg(md.name));

        md.axis = s.value(QString("axis"),QwtPlot::yLeft).value<QwtPlot::Axis>();
        md.plotIndex = s.value(QString("plotIndex"),d_plotCurves.size()).toInt() % d_allPlots.size();
        md.isVisible = s.value(QString("isVisible"),true).toBool();

        c->setAxes(QwtPlot::xBottom,md.axis);
        c->setVisible(md.isVisible);

        QColor color = s.value(QString("color"),palette().color(QPalette::Text)).value<QColor>();
        c->setPen(color);
        c->attach(d_allPlots.at(md.plotIndex));

        s.endGroup();

        md.min = value;
        md.max = value;

        d_plotCurves.append(md);
        d_allPlots.at(md.plotIndex)->replot();

    }
}


void TrackingViewWidget::moveCurveToPlot(int curveIndex, int newPlotIndex)
{
    if(curveIndex < 0 || curveIndex >= d_plotCurves.size() || newPlotIndex < 0 || newPlotIndex >= d_allPlots.size())
        return;

    //note which plot we're starting from, and move curve
    int oldPlotIndex = d_plotCurves.at(curveIndex).plotIndex;
    d_plotCurves.at(curveIndex).curve->detach();
    d_plotCurves.at(curveIndex).curve->attach(d_allPlots.at(newPlotIndex));

    //update metadata, and update autoscale parameters for plots
    d_plotCurves[curveIndex].plotIndex = newPlotIndex;

    //update settings, and replot
    if(!d_viewMode)
    {
        QSettings s;
        s.setValue(QString("trackingWidget/curves/%1/plotIndex").arg(d_plotCurves.at(curveIndex).name),
                   newPlotIndex);
        s.sync();
    }
    d_allPlots.at(oldPlotIndex)->replot();
    d_allPlots.at(newPlotIndex)->replot();
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
    {
        QSettings s;
        s.setValue(QString("trackingWidget/numPlots"),newNum);
        s.sync();
    }

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
                int newPlotIndex = d_plotCurves.at(i).plotIndex % newNum;
                moveCurveToPlot(i,newPlotIndex);
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
    QString name = QString("TrackingPlot%1").arg(d_allPlots.size());
    if(d_viewMode)
        name = QString("TrackingPlotView%1").arg(d_allPlots.size());

    TrackingPlot *tp = new TrackingPlot(d_name + BC::Key::plot + QString::number(d_allPlots.size()),this);

    tp->setMaxIndex(get<int>(BC::Key::numPlots,1)-1);

//    tp->setMinimumHeight(200);
//    tp->setMinimumWidth(375);
    tp->installEventFilter(this);


    int newPlotIndex = d_allPlots.size();
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
        p_gridLayout->setColumnStretch(0,1);
        p_gridLayout->setColumnStretch(1,1);
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
        p_gridLayout->addWidget(d_allPlots[3],1,0,1,2);
        p_gridLayout->addWidget(d_allPlots[4],1,3,1,2);
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
