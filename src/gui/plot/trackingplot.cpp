#include "trackingplot.h"

#include <QMouseEvent>
#include <QMenu>

#include <qwt6/qwt_legend_label.h>
#include <qwt6/qwt_legend.h>
#include <qwt6/qwt_plot_curve.h>

#include <gui/plot/customtracker.h>

TrackingPlot::TrackingPlot(QString name, QWidget *parent) : ZoomPanPlot(name, parent)
{
    QwtLegend *l = new QwtLegend;
    insertLegend(l,QwtPlot::BottomLegend);

    setAxisScaleDraw(QwtPlot::xBottom,new TimeScaleDraw);
    setAxisScaleDraw(QwtPlot::xTop,new TimeScaleDraw);

    p_tracker->setHorizontalTimeAxis(true);

}

TrackingPlot::~TrackingPlot()
{

}

void TrackingPlot::buildContextMenu(QMouseEvent *me)
{
    QMenu *menu = contextMenu();

    QAction *pushAction = menu->addAction(QString("Push X Axis"));
    pushAction->setToolTip(QString("Set the X axis ranges of all other plots to the range of this plot."));
    connect(pushAction,&QAction::triggered,this,&TrackingPlot::axisPushRequested);

    QAction *asAllAction = menu->addAction(QString("Autoscale All"));
    connect(asAllAction,&QAction::triggered,this,&TrackingPlot::autoScaleAllRequested);

    menu->popup(me->globalPos());
}
