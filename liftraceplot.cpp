#include "liftraceplot.h"

#include <QSettings>
#include <QMenu>
#include <QColorDialog>
#include <QMouseEvent>
#include <QWidgetAction>
#include <QSpinBox>
#include <QFormLayout>

#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_plot_zoneitem.h>
#include <qwt6/qwt_legend.h>
#include <qwt6/qwt_legend_label.h>
#include <qwt6/qwt_plot_textlabel.h>


LifTracePlot::LifTracePlot(QWidget *parent) :
    ZoomPanPlot(QString("lifTrace"),parent), d_numAverages(10), d_resetNext(true),
    d_lifGateMode(false), d_refGateMode(false)
{
    setAxisFont(QwtPlot::xBottom,QFont(QString("sans-serif"),8));
    setAxisFont(QwtPlot::yLeft,QFont(QString("sans-serif"),8));

    //build axis titles with small font. The <html> etc. tags are needed to display the mu character
    QwtText blabel(QString("Time (ns)"));
    blabel.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::xBottom,blabel);

    QwtText llabel(QString("LIF (V)"));
    llabel.setFont(QFont(QString("sans-serif"),8));
    this->setAxisTitle(QwtPlot::yLeft,llabel);

    QSettings s;
    QColor lifColor = s.value(QString("lifTracePlot/lifColor"),
                              QPalette().color(QPalette::Text)).value<QColor>();
    QColor refColor = s.value(QString("lifTracePlot/refColor"),
                              QPalette().color(QPalette::Text)).value<QColor>();

    p_integralLabel = new QwtPlotTextLabel();
    p_integralLabel->setZ(10.0);
    p_integralLabel->attach(this);

    p_lif = new QwtPlotCurve(QString("LIF"));
    p_lif->setRenderHint(QwtPlotItem::RenderAntialiased);
    p_lif->setPen(QPen(lifColor));
    p_lif->attach(this);
    p_lif->setZ(1.0);

    p_ref = new QwtPlotCurve(QString("Ref"));
    p_ref->setRenderHint(QwtPlotItem::RenderAntialiased);
    p_ref->setPen(QPen(refColor));
    p_ref->setZ(1.0);

    p_lifZone = new QwtPlotZoneItem();
    p_lifZone->setPen(QPen(lifColor,2.0));
    lifColor.setAlpha(75);
    p_lifZone->setBrush(QBrush(lifColor));
    p_lifZone->setOrientation(Qt::Vertical);
    p_lifZone->setZ(2.0);

    p_refZone = new QwtPlotZoneItem();
    p_refZone->setPen(QPen(refColor,2.0));
    refColor.setAlpha(75);
    p_refZone->setBrush(QBrush(refColor));
    p_refZone->setOrientation(Qt::Vertical);
    p_refZone->setZ(2.0);

    QwtLegend *leg = new QwtLegend(this);
    leg->contentsWidget()->installEventFilter(this);
    connect(leg,&QwtLegend::checked,this,&LifTracePlot::legendItemClicked);
    insertLegend(leg);

    initializeLabel(p_lif,true);

    connect(this,&LifTracePlot::plotRightClicked,this,&LifTracePlot::buildContextMenu);

    d_lifZoneRange.first = -1;
    d_lifZoneRange.second = -1;
    d_refZoneRange.first = -1;
    d_refZoneRange.second = -1;

    connect(this,&LifTracePlot::integralUpdate,this,&LifTracePlot::setIntegralText);
}

LifTracePlot::~LifTracePlot()
{
    p_lif->detach();
    delete p_lif;

    p_ref->detach();
    delete p_ref;

    p_lifZone->detach();
    delete p_lifZone;

    p_refZone->detach();
    delete p_refZone;

    p_integralLabel->detach();
    delete p_integralLabel;
}

void LifTracePlot::setNumAverages(int n)
{
    d_numAverages = n;
}

void LifTracePlot::newTrace(const LifTrace t)
{
    if(t.size() == 0)
        return;

    if(d_resetNext || d_currentTrace.size() == 0)
    {
        d_resetNext = false;
        traceProcessed(t);
    }
    else
    {
        d_currentTrace.rollAvg(t,d_numAverages);
        traceProcessed(d_currentTrace);
    }

}

void LifTracePlot::traceProcessed(const LifTrace t)
{
    d_currentTrace = t;

    if(t.hasRefData())
        emit integralUpdate(t.integrate(d_lifZoneRange.first,d_lifZoneRange.second,d_refZoneRange.first,d_refZoneRange.second));
    else
        emit integralUpdate(t.integrate(d_lifZoneRange.first,d_lifZoneRange.second));

    bool updateLif = false, updateRef = false;

    if(d_lifZoneRange.first < 0 || d_lifZoneRange.first >= t.size())
    {
        d_lifZoneRange.first = 0;
        updateLif = true;
    }
    if(d_lifZoneRange.second < d_lifZoneRange.first || d_lifZoneRange.second >= t.size()-1)
    {
        d_lifZoneRange.second = t.size()-1;
        updateLif = true;
    }
    if(t.hasRefData())
    {
        if(d_refZoneRange.first < 0 || d_refZoneRange.first >= t.size())
        {
            d_refZoneRange.first = 0;
            updateRef = true;
        }
        if(d_refZoneRange.second < d_refZoneRange.first || d_refZoneRange.second >= t.size()-1)
        {
            d_refZoneRange.second = t.size()-1;
            updateRef = true;
        }
    }

    setAxisAutoScaleRange(QwtPlot::xBottom,0.0,
                          d_currentTrace.spacing()*static_cast<double>(d_currentTrace.size())*1e9);

    if(updateLif)
        updateLifZone();

    if(p_lifZone->plot() != this)
        p_lifZone->attach(this);

    if(t.hasRefData())
    {
        if(updateRef)
            updateRefZone();

        if(p_ref->plot() != this)
        {
            p_ref->attach(this);
            initializeLabel(p_ref,true);
        }

        if(p_refZone->plot() != this)
            p_refZone->attach(this);
    }
    else
    {
        if(p_ref->plot() == this)
            p_ref->detach();
        if(p_refZone->plot() == this)
            p_refZone->detach();
    }

    filterData();
    replot();
}

void LifTracePlot::buildContextMenu(QMouseEvent *me)
{
    QMenu *m = contextMenu();

    QAction *lifZoneAction = m->addAction(QString("Change LIF Gate..."));
    connect(lifZoneAction,&QAction::triggered,this,&LifTracePlot::changeLifGateRange);
    if(d_currentTrace.size() == 0 || !p_lifZone->isVisible() || !isEnabled())
        lifZoneAction->setEnabled(false);


    QAction *refZoneAction = m->addAction(QString("Change Ref Gate..."));
    connect(refZoneAction,&QAction::triggered,this,&LifTracePlot::changeRefGateRange);
    if(!d_currentTrace.hasRefData() || !p_refZone->isVisible() || !isEnabled())
        refZoneAction->setEnabled(false);

    m->addSeparator();

    QAction *resetAction = m->addAction(QString("Reset Averages"));
    connect(resetAction,&QAction::triggered,this,&LifTracePlot::reset);
    if(d_currentTrace.size() == 0 || !isEnabled())
        resetAction->setEnabled(false);

    QWidgetAction *wa = new QWidgetAction(m);
    QWidget *w = new QWidget(m);
    QSpinBox *shotsBox = new QSpinBox(w);
    QFormLayout *fl = new QFormLayout();

    fl->addRow(QString("Average"),shotsBox);

    shotsBox->setRange(1,__INT32_MAX__);
    shotsBox->setSingleStep(10);
    shotsBox->setValue(d_numAverages);
    shotsBox->setSuffix(QString(" shots"));
    connect(shotsBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),
            [=](int n){ d_numAverages = n; });
    if(!isEnabled())
        shotsBox->setEnabled(false);

    w->setLayout(fl);
    wa->setDefaultWidget(w);
    m->addAction(wa);


    m->popup(me->globalPos());
}

void LifTracePlot::changeLifColor()
{
    QColor currentColor = p_lif->pen().color();

    QColor newColor = QColorDialog::getColor(currentColor,this,QString("Choose New LIF Color"));
    if(!newColor.isValid())
        return;

    QSettings s;
    s.setValue(QString("lifTracePlot/lifColor"),newColor);

    p_lif->setPen(QPen(newColor));
    p_lifZone->setPen(QPen(newColor,2.0));
    newColor.setAlpha(75);
    p_lifZone->setBrush(newColor);

    initializeLabel(p_lif,p_lif->isVisible());

    replot();
}

void LifTracePlot::changeRefColor()
{
    QColor currentColor = p_ref->pen().color();

    QColor newColor = QColorDialog::getColor(currentColor,this,QString("Choose New Reference Color"));
    if(!newColor.isValid())
        return;

    QSettings s;
    s.setValue(QString("lifTracePlot/refColor"),newColor);

    p_ref->setPen(QPen(newColor));
    p_refZone->setPen(QPen(newColor,2.0));
    newColor.setAlpha(75);
    p_refZone->setBrush(newColor);

    initializeLabel(p_ref,p_ref->isVisible());

    if(p_ref->plot() == this)
        replot();
}

void LifTracePlot::legendItemClicked(QVariant info, bool checked, int index)
{
    Q_UNUSED(index);

    QwtPlotCurve *c = dynamic_cast<QwtPlotCurve*>(infoToItem(info));
    if(c == nullptr)
        return;

    c->setVisible(checked);
    if(c == p_lif)
        p_lifZone->setVisible(checked);
    else if(c == p_ref)
        p_refZone->setVisible(checked);

    replot();
}

void LifTracePlot::reset()
{
    d_resetNext = true;
}

void LifTracePlot::setIntegralText(double d)
{
    QwtText t;
    QString text = QString::number(d,'e',3);

    t.setRenderFlags(Qt::AlignRight | Qt::AlignTop);
    t.setText(text);
    t.setBackgroundBrush(QBrush(QPalette().color(QPalette::Window)));
    QColor border = QPalette().color(QPalette::Text);
    border.setAlpha(0);
    t.setBorderPen(QPen(border));
    t.setColor(QPalette().color(QPalette::Text));

    QFont f(QString("monospace"),14);
    f.setBold(true);
    t.setFont(f);

    p_integralLabel->setText(t);
}

void LifTracePlot::changeLifGateRange()
{
    d_lifGateMode = true;
    canvas()->setMouseTracking(true);
}

void LifTracePlot::changeRefGateRange()
{
    d_refGateMode = true;
    canvas()->setMouseTracking(true);
}

void LifTracePlot::initializeLabel(QwtPlotCurve *curve, bool isVisible)
{
    QwtLegendLabel* item = static_cast<QwtLegendLabel*>
            (static_cast<QwtLegend*>(legend())->legendWidget(itemToInfo(curve)));

    item->setItemMode(QwtLegendData::Checkable);
    item->setChecked(isVisible);
}

void LifTracePlot::updateLifZone()
{
    double x1 = static_cast<double>(d_lifZoneRange.first)*d_currentTrace.spacing()*1e9;
    double x2 = static_cast<double>(d_lifZoneRange.second)*d_currentTrace.spacing()*1e9;
    p_lifZone->setInterval(x1,x2);

    if(d_currentTrace.hasRefData())
        emit integralUpdate(d_currentTrace.integrate(d_lifZoneRange.first,d_lifZoneRange.second,d_refZoneRange.first,d_refZoneRange.second));
    else
        emit integralUpdate(d_currentTrace.integrate(d_lifZoneRange.first,d_lifZoneRange.second));
}

void LifTracePlot::updateRefZone()
{
    double x1 = static_cast<double>(d_refZoneRange.first)*d_currentTrace.spacing()*1e9;
    double x2 = static_cast<double>(d_refZoneRange.second)*d_currentTrace.spacing()*1e9;
    p_refZone->setInterval(x1,x2);

    emit integralUpdate(d_currentTrace.integrate(d_lifZoneRange.first,d_lifZoneRange.second,d_refZoneRange.first,d_refZoneRange.second));
}

void LifTracePlot::filterData()
{
    if(d_currentTrace.size() < 2)
        return;

    QVector<QPointF> lifData = d_currentTrace.lifToXY();
    QVector<QPointF> refData;
    if(d_currentTrace.hasRefData())
        refData = d_currentTrace.refToXY();


    double firstPixel = 0.0;
    double lastPixel = canvas()->width();
    QwtScaleMap map = canvasMap(QwtPlot::xBottom);

    QVector<QPointF> lifFiltered, refFiltered;
    double yMin = 0.0, yMax = 0.0;

    //find first data point that is in the range of the plot
    //note: x data for lif and ref are assumed to be the same!
    int dataIndex = 0;
    while(dataIndex+1 < lifData.size() && map.transform(lifData.at(dataIndex).x()*1e9) < firstPixel)
        dataIndex++;

    //add the previous point to the filtered array
    //this will make sure the curve always goes to the edge of the plot
    if(dataIndex-1 >= 0)
    {
        lifFiltered.append(lifData.at(dataIndex-1));
        if(d_currentTrace.hasRefData())
            refFiltered.append(refData.at(dataIndex-1));
    }

    //at this point, dataIndex is at the first point within the range of the plot. loop over pixels, compressing data
    for(double pixel = firstPixel; pixel<lastPixel; pixel+=1.0)
    {
        double lifMin = lifData.at(dataIndex).y(), lifMax = lifMin;
        double refMin = 0.0, refMax = 0.0;
        if(d_currentTrace.hasRefData())
        {
            refMin = refData.at(dataIndex).y();
            refMax = refMin;
        }

        int lifMinIndex = dataIndex, lifMaxIndex = dataIndex, refMinIndex = dataIndex, refMaxIndex = dataIndex;
        int numPnts = 0;

        while(dataIndex+1 < lifData.size() && map.transform(lifData.at(dataIndex).x()*1e9) < pixel+1.0)
        {
            if(lifData.at(dataIndex).y() < lifMin)
            {
                lifMin = lifData.at(dataIndex).y();
                lifMinIndex = dataIndex;
            }
            if(lifData.at(dataIndex).y() > lifMax)
            {
                lifMax = lifData.at(dataIndex).y();
                lifMaxIndex = dataIndex;
            }
            if(d_currentTrace.hasRefData())
            {
                if(refData.at(dataIndex).y() < refMin)
                {
                    refMin = refData.at(dataIndex).y();
                    refMinIndex = dataIndex;
                }
                if(refData.at(dataIndex).y() > refMax)
                {
                    refMax = refData.at(dataIndex).y();
                    refMaxIndex = dataIndex;
                }
            }

            dataIndex++;
            numPnts++;
        }
        if(lifFiltered.isEmpty())
        {
            yMin = lifMin;
            yMax = lifMax;
            if(d_currentTrace.hasRefData())
            {
                yMin = qMin(lifMin,refMin);
                yMax = qMax(lifMax,refMax);
            }
        }
        else
        {
            yMin = qMin(lifMin,yMin);
            yMax = qMax(lifMax,yMax);
            if(d_currentTrace.hasRefData())
            {
                yMin = qMin(yMin,refMin);
                yMax = qMax(yMax,refMax);
            }
        }
        if(numPnts == 1)
        {
            lifFiltered.append(QPointF(lifData.at(dataIndex-1).x()*1e9,lifData.at(dataIndex-1).y()));
            if(d_currentTrace.hasRefData())
                refFiltered.append(QPointF(lifData.at(dataIndex-1).x()*1e9,refData.at(dataIndex-1).y()));
        }
        else if (numPnts > 1)
        {
            QPointF first(map.invTransform(pixel),lifData.at(lifMinIndex).y());
            QPointF second(map.invTransform(pixel),lifData.at(lifMaxIndex).y());
            lifFiltered.append(first);
            lifFiltered.append(second);
            if(d_currentTrace.hasRefData())
            {
                QPointF refFirst(map.invTransform(pixel),refData.at(refMinIndex).y());
                QPointF refSecond(map.invTransform(pixel),refData.at(refMaxIndex).y());
                refFiltered.append(refFirst);
                refFiltered.append(refSecond);
            }
        }
    }

    if(dataIndex < lifData.size())
    {
        QPointF p = lifData.at(dataIndex);
        p.setX(p.x()*1e9);
        lifFiltered.append(p);
        if(d_currentTrace.hasRefData())
        {
            p = refData.at(dataIndex);
            p.setX(p.x()*1e9);
            refFiltered.append(p);
        }
    }

    setAxisAutoScaleRange(QwtPlot::yLeft,yMin,yMax);
    //assign data to curve object
    p_lif->setSamples(lifFiltered);
    if(d_currentTrace.hasRefData())
        p_ref->setSamples(refFiltered);

}

bool LifTracePlot::eventFilter(QObject *obj, QEvent *ev)
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
                        if(c == p_lif)
                        {
                            changeLifColor();
                            ev->accept();
                            return true;
                        }
                        else if(c == p_ref)
                        {
                            changeRefColor();
                            ev->accept();
                            return true;
                        }
                    }
                }
            }
        }

        if(d_lifGateMode)
        {
            d_lifGateMode = false;
            canvas()->setMouseTracking(false);
            ev->accept();
            return true;
        }

        if(d_refGateMode)
        {
            d_refGateMode = false;
            canvas()->setMouseTracking(false);
            ev->accept();
            return true;
        }
    }
    else if(ev->type() == QEvent::Wheel)
    {
        QWheelEvent *we = static_cast<QWheelEvent*>(ev);
        int d = we->angleDelta().y()/120;

        if(we->modifiers() & Qt::ControlModifier)
            d*=5;

        if(d_lifGateMode)
        {
            int newSpacing = d_lifZoneRange.second - d_lifZoneRange.first + 2*d;
            if(newSpacing > 3)
            {
                d_lifZoneRange.first = qBound(0,d_lifZoneRange.first-d,qMin(d_lifZoneRange.second+d-1,d_currentTrace.size()-1));
                d_lifZoneRange.second = qBound(d_lifZoneRange.first,d_lifZoneRange.second+d,d_currentTrace.size()-1);
                updateLifZone();
                replot();
                ev->accept();
                return true;
            }
            else
            {
                ev->ignore();
                return true;
            }
        }

        if(d_refGateMode)
        {
            int newSpacing = d_refZoneRange.second - d_refZoneRange.first + 2*d;
            if(newSpacing > 3)
            {
                d_refZoneRange.first = qBound(0,d_refZoneRange.first-d,qMin(d_refZoneRange.second+d-1,d_currentTrace.size()-1));
                d_refZoneRange.second = qBound(d_refZoneRange.first,d_refZoneRange.second+d,d_currentTrace.size()-1);
                updateRefZone();
                replot();
                ev->accept();
                return true;
            }
            else
            {
                ev->ignore();
                return true;
            }
        }
    }
    else if(ev->type() == QEvent::MouseMove)
    {
        QMouseEvent *me = static_cast<QMouseEvent*>(ev);
        double mousePos = canvasMap(QwtPlot::xBottom).invTransform(me->localPos().x());
        int newCenter = static_cast<int>(round(mousePos/(d_currentTrace.spacing()*1e9)));

        if(d_lifGateMode)
        {
            //preserve spacing, move center
            int oldCenter = (d_lifZoneRange.second + d_lifZoneRange.first)/2;
            int shift = newCenter - oldCenter;
            if(d_lifZoneRange.first + shift >= 0 && d_lifZoneRange.second + shift < d_currentTrace.size())
            {
                d_lifZoneRange.first += shift;
                d_lifZoneRange.second += shift;

                updateLifZone();
                replot();
                ev->accept();
                return true;
            }
        }

        if(d_refGateMode)
        {
            //preserve spacing, move center
            int oldCenter = (d_refZoneRange.second + d_refZoneRange.first)/2;
            int shift = newCenter - oldCenter;
            if(d_refZoneRange.first + shift >= 0 && d_refZoneRange.second + shift < d_currentTrace.size())
            {
                d_refZoneRange.first += shift;
                d_refZoneRange.second += shift;

                updateRefZone();
                replot();
                ev->accept();
                return true;
            }
        }
    }

    return ZoomPanPlot::eventFilter(obj,ev);
}
