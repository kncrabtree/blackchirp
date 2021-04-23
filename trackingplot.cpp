#include "trackingplot.h"

#include <QMouseEvent>
#include <QMenu>

#include <qwt6/qwt_legend_label.h>
#include <qwt6/qwt_legend.h>
#include <qwt6/qwt_plot_curve.h>

TrackingPlot::TrackingPlot(QString name, QWidget *parent) : ZoomPanPlot(name, parent)
{
    QwtLegend *l = new QwtLegend;
    l->contentsWidget()->installEventFilter(this);
    connect(l,&QwtLegend::checked,this,&TrackingPlot::legendItemClicked);
    insertLegend(l,QwtPlot::BottomLegend);

    setAxisScaleDraw(QwtPlot::xBottom,new TimeScaleDraw);
    setAxisScaleDraw(QwtPlot::xTop,new TimeScaleDraw);

    setAxisFont(QwtPlot::xBottom,QFont(QString("sans-serif"),8));
    setAxisFont(QwtPlot::xTop,QFont(QString("sans-serif"),8));
    setAxisFont(QwtPlot::yLeft,QFont(QString("sans-serif"),8));
    setAxisFont(QwtPlot::yRight,QFont(QString("sans-serif"),8));

    canvas()->installEventFilter(this);
}

TrackingPlot::~TrackingPlot()
{

}

void TrackingPlot::initializeLabel(QwtPlotCurve *curve, bool isVisible)
{
    QwtLegendLabel* item = static_cast<QwtLegendLabel*>
            (static_cast<QwtLegend*>(legend())->legendWidget(itemToInfo(curve)));

    item->setItemMode(QwtLegendData::Checkable);
    item->setChecked(isVisible);
}

void TrackingPlot::legendItemClicked(QVariant info, bool checked, int index)
{
    Q_UNUSED(index);

    QwtPlotCurve *c = dynamic_cast<QwtPlotCurve*>(infoToItem(info));
    if(c != nullptr)
        emit curveVisiblityToggled(c,checked);
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

void TrackingPlot::filterData()
{
}

bool TrackingPlot::eventFilter(QObject *obj, QEvent *ev)
{
    if(ev->type() == QEvent::MouseButtonPress)
    {
        QwtLegend *l = static_cast<QwtLegend*>(legend());
        if(obj == l->contentsWidget())
        {
            QMouseEvent *me = dynamic_cast<QMouseEvent*>(ev);
            if(me != nullptr)
            {
                QwtLegendLabel *ll = dynamic_cast<QwtLegendLabel*>(l->contentsWidget()->childAt(me->pos()));
                if(ll != nullptr)
                {
                    QVariant item = l->itemInfo(ll);
                    QwtPlotCurve *c = dynamic_cast<QwtPlotCurve*>(infoToItem(item));
                    if(c == nullptr)
                    {
                        ev->ignore();
                        return true;
                    }

                    if(me->button() == Qt::RightButton)
                    {
                        emit legendItemRightClicked(c,me);
                        ev->accept();
                        return true;
                    }
                }
            }
        }
    }

    return ZoomPanPlot::eventFilter(obj,ev);
}

