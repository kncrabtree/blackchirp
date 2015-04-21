#include "led.h"

#include <QPainter>
#include <QPalette>
#include <QRadialGradient>

Led::Led(QWidget *parent) : QWidget(parent), d_ledOn(false)
{
    setLedSize(15);
    d_onColor = QColor(0,175,0);
    d_offColor = QColor(0,30,0);
}

Led::~Led()
{

}

void Led::setLedSize(int newSize)
{
    d_diameter = newSize;
    setFixedSize(d_diameter+5,d_diameter+5);
    update();
}

void Led::setState(bool on)
{
    d_ledOn = on;
    update();
}



void Led::paintEvent(QPaintEvent *ev)
{
    Q_UNUSED(ev)

    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    int offset = (width()-d_diameter)/2;

    QRadialGradient g(offset+d_diameter/2,offset+d_diameter/2,35*d_diameter/100,2*d_diameter/5,2*d_diameter/5);

    if(d_ledOn)
    {
        g.setColorAt(0,QColor(255,255,255,200));
        g.setColorAt(1,d_onColor);
        p.setPen(d_onColor);
    }
    else
    {
        g.setColorAt(0,QColor(220,220,220,150));
        g.setColorAt(1,d_offColor);
        p.setPen(d_offColor);
    }

    p.setBrush(QBrush(g));
    p.drawEllipse(QPoint(width()/2,height()/2),d_diameter/2,d_diameter/2);

}
