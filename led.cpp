#include "led.h"
#include <QPainter>
#include <QPalette>
#include <QRadialGradient>

Led::Led(QWidget *parent) : QWidget(parent), d_ledOn(false)
{
    setLedSize(15);
    d_onColor = QColor(0,175,0);
//    d_offColor = QPalette().color(QPalette::Base);
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

    QRadialGradient g(d_diameter/2,d_diameter/2,d_diameter/3,-d_diameter/3,d_diameter/8);
    g.setColorAt(0,QColor(255,255,255,128));


    if(d_ledOn)
        g.setColorAt(1,d_onColor);
//        p.setBrush(QBrush(d_onColor));
    else
        g.setColorAt(1,d_offColor);
//        p.setBrush(QBrush(d_offColor));

    p.setBrush(QBrush(g));
    p.drawEllipse(0,0,d_diameter,d_diameter);

}
