#include "liftraceplot.h"

#include <QApplication>
#include <QMenu>
#include <QColorDialog>
#include <QMouseEvent>
#include <QWidgetAction>
#include <QSpinBox>
#include <QFormLayout>
#include <QFileDialog>
#include <QMessageBox>

#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_plot_zoneitem.h>
#include <qwt6/qwt_legend.h>
#include <qwt6/qwt_legend_label.h>
#include <qwt6/qwt_plot_textlabel.h>

#include <src/gui/plot/blackchirpplotcurve.h>


LifTracePlot::LifTracePlot(QWidget *parent) :
    ZoomPanPlot(BC::Key::lifTracePlot,parent), d_resetNext(true),
    d_lifGateMode(false), d_refGateMode(false), d_displayOnly(false)
{

    setPlotAxisTitle(QwtPlot::xBottom,QString("Time (ns)"));
    setPlotAxisTitle(QwtPlot::yLeft,QString("LIF (V)"));

    p_integralLabel = new QwtPlotTextLabel();
    p_integralLabel->setZ(10.0);
    p_integralLabel->attach(this);
    p_integralLabel->setItemAttribute(QwtPlotItem::AutoScale,false);

    p_lif = new BlackchirpPlotCurve(BC::Key::lifCurve);
    p_lif->setZ(1.0);

    p_ref = new BlackchirpPlotCurve(BC::Key::refCurve);
    p_ref->setZ(1.0);

    p_lifZone = new QwtPlotZoneItem();
    p_lifZone->setOrientation(Qt::Vertical);
    p_lifZone->setZ(2.0);
    p_lifZone->setItemAttribute(QwtPlotItem::AutoScale,false);



    p_refZone = new QwtPlotZoneItem();
    p_refZone->setOrientation(Qt::Vertical);
    p_refZone->setZ(2.0);
    p_refZone->setItemAttribute(QwtPlotItem::AutoScale,false);


    insertLegend( new QwtLegend(this),QwtPlot::BottomLegend);

    QSettings s2(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    d_lifZoneRange.first = s2.value(QString("lifConfig/lifStart"),-1).toInt();
    d_lifZoneRange.second = s2.value(QString("lifConfig/lifEnd"),-1).toInt();
    d_refZoneRange.first = s2.value(QString("lifConfig/refStart"),-1).toInt();
    d_refZoneRange.second = s2.value(QString("lifConfig/refEnd"),-1).toInt();
    d_numAverages = s2.value(QString("lifConfig/numAverages"),10).toInt();

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

void LifTracePlot::setLifGateRange(int begin, int end)
{
    d_lifZoneRange.first = begin;
    d_lifZoneRange.second = end;
    updateLifZone();
}

void LifTracePlot::setRefGateRange(int begin, int end)
{
    d_refZoneRange.first = begin;
    d_refZoneRange.second = end;
    updateRefZone();
}

LifConfig LifTracePlot::getSettings(LifConfig c)
{
    c.setLifGate(d_lifZoneRange.first,d_lifZoneRange.second);
    c.setRefGate(d_refZoneRange.first,d_refZoneRange.second);
    c.setShotsPerPoint(d_numAverages);

    return c;
}

void LifTracePlot::setNumAverages(int n)
{
    d_numAverages = n;
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.setValue(QString("lifConfig/numAverages"),n);
    s.sync();
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
    bool updateLif = false, updateRef = false;
    if(d_currentTrace.size() == 0)
    {
        updateLif = true;
        if(t.hasRefData())
            updateRef = true;
    }

    d_currentTrace = t;
    p_lif->setCurveData(t.lifToXY());
    p_ref->setCurveData(t.refToXY());

    if(t.hasRefData())
        emit integralUpdate(t.integrate(d_lifZoneRange.first,d_lifZoneRange.second,d_refZoneRange.first,d_refZoneRange.second));
    else
        emit integralUpdate(t.integrate(d_lifZoneRange.first,d_lifZoneRange.second));



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

    if(p_lif->plot() != this)
        p_lif->attach(this);

    if(updateLif)
        updateLifZone();

    if(p_lifZone->plot() != this)
        p_lifZone->attach(this);

    if(t.hasRefData())
    {
        if(updateRef)
            updateRefZone();

        if(p_ref->plot() != this)
            p_ref->attach(this);

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

    replot();
}

void LifTracePlot::buildContextMenu(QMouseEvent *me)
{
    QMenu *m = contextMenu();
    QAction *exportAction=m->addAction(QString("Export XY..."));
    connect(exportAction,&QAction::triggered,this,&LifTracePlot::exportXY);

    QAction *lifZoneAction = m->addAction(QString("Change LIF Gate..."));
    connect(lifZoneAction,&QAction::triggered,this,&LifTracePlot::changeLifGateRange);
    if(d_currentTrace.size() == 0 || !p_lifZone->isVisible() || !isEnabled())
        lifZoneAction->setEnabled(false);


    QAction *refZoneAction = m->addAction(QString("Change Ref Gate..."));
    connect(refZoneAction,&QAction::triggered,this,&LifTracePlot::changeRefGateRange);
    if(!d_currentTrace.hasRefData() || !p_refZone->isVisible() || !isEnabled())
        refZoneAction->setEnabled(false);

    if(!d_displayOnly)
    {

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
        connect(shotsBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,&LifTracePlot::setNumAverages);
        if(!isEnabled())
            shotsBox->setEnabled(false);

        w->setLayout(fl);
        wa->setDefaultWidget(w);
        m->addAction(wa);
    }


    m->popup(me->globalPos());
}

void LifTracePlot::checkColors()
{

    p_lif->updateFromSettings();
    auto lc = p_lif->pen().color();
    p_lifZone->setPen(QPen(lc,2.0));
    lc.setAlpha(75);
    p_lifZone->setBrush(QBrush(lc));

    p_ref->updateFromSettings();
    auto rc = p_ref->pen().color();
    p_refZone->setPen(QPen(rc,2.0));
    rc.setAlpha(75);
    p_refZone->setBrush(QBrush(rc));

    ZoomPanPlot::replot();



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

void LifTracePlot::clearPlot()
{
    p_lif->detach();
    p_lifZone->detach();
    p_ref->detach();
    p_refZone->detach();
    p_integralLabel->setText(QString(""));

    d_currentTrace = LifTrace();

    replot();
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

bool LifTracePlot::eventFilter(QObject *obj, QEvent *ev)
{
    if(ev->type() == QEvent::MouseButtonPress)
    {
        if(d_lifGateMode)
        {
            d_lifGateMode = false;
            canvas()->setMouseTracking(false);

            QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
            s.setValue(QString("lifConfig/lifStart"),d_lifZoneRange.first);
            s.setValue(QString("lifConfig/lifEnd"),d_lifZoneRange.second);
            emit lifGateUpdated(d_lifZoneRange.first,d_lifZoneRange.second);
            ev->accept();
            return true;
        }

        if(d_refGateMode)
        {
            d_refGateMode = false;
            canvas()->setMouseTracking(false);

            QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
            s.setValue(QString("lifConfig/refStart"),d_refZoneRange.first);
            s.setValue(QString("lifConfig/refEnd"),d_refZoneRange.second);
            emit refGateUpdated(d_refZoneRange.first,d_refZoneRange.second);
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

void LifTracePlot::replot()
{
    //this function calls ZoomPanPlot::replot()
    checkColors();
}

void LifTracePlot::exportXY()
{
    QString path = BlackChirp::getExportDir();
    QString name = QFileDialog::getSaveFileName(this,QString("Export LIF Trace"),path + QString("/lifxy.txt"));
    if(name.isEmpty())
        return;
    QFile f(name);
    if(!f.open(QIODevice::WriteOnly))
    {
        QMessageBox::critical(this,QString("Export Failed"),QString("Could not open file %1 for writing. Please choose a different filename.").arg(name));
        return;
    }
    QApplication::setOverrideCursor(Qt::BusyCursor);
    f.write(QString("time\tlif").toLatin1());
    auto d = d_currentTrace.lifToXY();
    for(int i=0;i<d.size();i++)
    {
        f.write(QString("\n%1\t%2").arg(d.at(i).x(),0,'e',6)
                    .arg(d.at(i).y(),0,'e',12).toLatin1());
    }
    f.close();
    QApplication::restoreOverrideCursor();
    BlackChirp::setExportDir(name);
}
