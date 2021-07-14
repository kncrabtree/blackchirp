#include <gui/plot/customtracker.h>

#include <qwt6/qwt_scale_map.h>
#include <qwt6/qwt_picker_machine.h>
#include <qwt6/qwt_date.h>

CustomTracker::CustomTracker(QWidget *canvas) : QwtPlotPicker(canvas)
{
    setStateMachine(new QwtPickerClickPointMachine);
    setMousePattern(QwtEventPattern::MouseSelect1,Qt::RightButton);
    setTrackerMode(QwtPicker::AlwaysOn);
    setTrackerPen(QPen(QPalette().color(QPalette::Text)));
}

QwtText CustomTracker::trackerText(const QPoint &pos) const
{
    if(plot() == nullptr)
        return QwtText();

    QPalette p;
    QColor bg( p.window().color() );
    bg.setAlpha( 200 );

    QStringList l;

    for(auto it = d_axes.constBegin(); it != d_axes.constEnd(); it++)
    {
        if(plot()->axisEnabled(it.key()))
        {
            auto map = plot()->canvasMap(it.key());;
            auto d = d_details[it.key()].decimals;
            auto s = d_details[it.key()].scientific;
            auto text = it.value()+QString(": %1");
            auto val = pos.y();
            if(it.key() == QwtPlot::xBottom || it.key() == QwtPlot::xTop)
                val = pos.x();
            if((it.key() == QwtPlot::xBottom || it.key() == QwtPlot::xTop) && d_hTime)
                l.append(text.arg(QwtDate::toDateTime(map.invTransform(val),Qt::LocalTime).toString(QString("M/d h:mm:ss"))));
            else if(s)
                l.append(text.arg(map.invTransform(val),0,'E',d));
            else
                l.append(text.arg(map.invTransform(val),0,'f',d));
        }
    }

    if(l.isEmpty())
        return QwtText();

    QwtText text = l.join(QString("\n"));
    text.setColor(p.text().color());
    text.setBackgroundBrush( QBrush( bg ) );
    return text;
}

void CustomTracker::setDecimals(QwtPlot::Axis axis, int decimals)
{
    d_details[axis].decimals = decimals;
}

void CustomTracker::setScientific(QwtPlot::Axis axis, bool scientific)
{
    d_details[axis].scientific = scientific;
}
