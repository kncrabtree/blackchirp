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

#include "trackingplot.h"
#include "datastructs.h"


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

void TrackingViewWidget::pointUpdated(const QList<QPair<QString, QVariant> > list, bool plot)
{
    if(!plot)
        return;

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
        QString realName = BlackChirp::channelNameLookup(md.name);
        if(realName.isEmpty())
            realName = md.name;

        QwtPlotCurve *c = new QwtPlotCurve(realName);
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

void TrackingViewWidget::curveVisibilityToggled(QwtPlotCurve *c, bool visible)
{
    int i = findCurveIndex(c);
    if(i < 0)
        return;

    QSettings s;
    s.setValue(QString("trackingWidget/curves/%1/isVisible").arg(d_plotCurves.at(i).name),visible);
    s.sync();

    d_plotCurves[i].isVisible = visible;
    d_plotCurves[i].curve->setVisible(visible);
    setAutoScaleYRanges(d_plotCurves.at(i).plotIndex,d_plotCurves.at(i).axis);
    d_allPlots.at(d_plotCurves.at(i).plotIndex)->replot();

}


void TrackingViewWidget::curveContextMenuRequested(QwtPlotCurve *c, QMouseEvent *me)
{
    if(c == nullptr || me == nullptr)
        return;

    int i = findCurveIndex(c);
    if(i<0)
        return;

    QMenu *menu = new QMenu();

    QAction *colorAction = menu->addAction(QString("Change color..."));
    connect(colorAction,&QAction::triggered,this, [=](){ changeCurveColor(i); } );

    QMenu *moveMenu = menu->addMenu(QString("Change plot"));
    QActionGroup *moveGroup = new QActionGroup(moveMenu);
    moveGroup->setExclusive(true);
    for(int j=0; j<d_allPlots.size(); j++)
    {
        QAction *a = moveGroup->addAction(QString("Move to plot %1").arg(j+1));
        a->setCheckable(true);
        if(j == d_plotCurves.at(i).plotIndex)
        {
            a->setEnabled(false);
            a->setChecked(true);
        }
        else
        {
            connect(a,&QAction::triggered,this, [=](){ moveCurveToPlot(i,j); });
            a->setChecked(false);
        }
    }
    moveMenu->addActions(moveGroup->actions());

    menu->addSection(QString("Axis"));
    QActionGroup *axisGroup = new QActionGroup(menu);
    axisGroup->setExclusive(true);
    QAction *lAction = axisGroup->addAction(QString("Left"));
    QAction *rAction = axisGroup->addAction(QString("Right"));
    lAction->setCheckable(true);
    rAction->setCheckable(true);
    if(d_plotCurves.at(i).axis == QwtPlot::yLeft)
    {
        lAction->setEnabled(false);
        lAction->setChecked(true);
        connect(rAction,&QAction::triggered,this,[=](){ changeCurveAxis(i); });
    }
    else
    {
        rAction->setEnabled(false);
        rAction->setChecked(true);
        connect(lAction,&QAction::triggered,this,[=](){ changeCurveAxis(i); });
    }
    menu->addActions(axisGroup->actions());

    connect(menu,&QMenu::aboutToHide,menu,&QObject::deleteLater);
    menu->popup(me->globalPos());
}

void TrackingViewWidget::changeCurveColor(int curveIndex)
{
    if(curveIndex < 0 || curveIndex >= d_plotCurves.size())
        return;

    QColor currentColor = d_plotCurves.at(curveIndex).curve->pen().color();
    QColor newColor = QColorDialog::getColor(currentColor,this,QString("Select New Color"));
    if(newColor.isValid())
    {
        d_plotCurves.at(curveIndex).curve->setPen(newColor);
        QSettings s;
        s.setValue(QString("trackingWidget/curves/%1/color").arg(d_plotCurves.at(curveIndex).name),newColor);
        s.sync();
        d_allPlots.at(d_plotCurves.at(curveIndex).plotIndex)->initializeLabel
                (d_plotCurves.at(curveIndex).curve,d_plotCurves.at(curveIndex).isVisible);
        d_allPlots.at(d_plotCurves.at(curveIndex).plotIndex)->replot();
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
    setAutoScaleYRanges(oldPlotIndex,d_plotCurves.at(curveIndex).axis);
    setAutoScaleYRanges(newPlotIndex,d_plotCurves.at(curveIndex).axis);

    //the new legend label needs to be made checkable
    d_allPlots.at(newPlotIndex)->initializeLabel
            (d_plotCurves.at(curveIndex).curve,d_plotCurves.at(curveIndex).isVisible);

    //update settings, and replot
    QSettings s;
    s.setValue(QString("trackingWidget/curves/%1/plotIndex").arg(d_plotCurves.at(curveIndex).name),
               newPlotIndex);
    s.sync();
    d_allPlots.at(oldPlotIndex)->replot();
    d_allPlots.at(newPlotIndex)->replot();
}

void TrackingViewWidget::changeCurveAxis(int curveIndex)
{
    if(curveIndex < 0 || curveIndex > d_plotCurves.size())
        return;

    QwtPlot::Axis oldAxis = d_plotCurves.at(curveIndex).axis;
    QwtPlot::Axis newAxis;
    if(oldAxis == QwtPlot::yLeft)
        newAxis = QwtPlot::yRight;
    else
        newAxis = QwtPlot::yLeft;

    d_plotCurves.at(curveIndex).curve->setAxes(QwtPlot::xBottom,newAxis);
    d_plotCurves[curveIndex].axis = newAxis;

    QSettings s;
    s.setValue(QString("trackingWidget/curves/%1/axis").arg(d_plotCurves.at(curveIndex).name),newAxis);
    s.sync();

    setAutoScaleYRanges(d_plotCurves.at(curveIndex).plotIndex,oldAxis);
    setAutoScaleYRanges(d_plotCurves.at(curveIndex).plotIndex,newAxis);
    d_allPlots.at(d_plotCurves.at(curveIndex).plotIndex)->replot();
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

    QSettings s;
    s.setValue(QString("trackingWidget/numPlots"),newNum);


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

    s.sync();

}

int TrackingViewWidget::findCurveIndex(QwtPlotCurve *c)
{
    if(c == nullptr)
        return -1;

    for(int i=0; i<d_plotCurves.size(); i++)
    {
        if(d_plotCurves.at(i).curve == c)
            return i;
    }

    return -1;
}

void TrackingViewWidget::addNewPlot()
{
    TrackingPlot *tp = new TrackingPlot(QString("TrackingPlot%1").arg(d_allPlots.size()),this);

    tp->setAxisAutoScaleRange(QwtPlot::xBottom,d_xRange.first,d_xRange.second);
    tp->setAxisAutoScaleRange(QwtPlot::xTop,d_xRange.first,d_xRange.second);

    tp->setMinimumHeight(200);
    tp->setMinimumWidth(375);
    tp->installEventFilter(this);

    //signal-slot connections go here
    connect(tp,&TrackingPlot::curveVisiblityToggled,this,&TrackingViewWidget::curveVisibilityToggled);
    connect(tp,&TrackingPlot::legendItemRightClicked,this,&TrackingViewWidget::curveContextMenuRequested);
    int newPlotIndex = d_allPlots.size();
    connect(tp,&TrackingPlot::axisPushRequested,this,[=](){ pushXAxis(newPlotIndex); });
    connect(tp,&TrackingPlot::autoScaleAllRequested,this,&TrackingViewWidget::autoScaleAll);

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
