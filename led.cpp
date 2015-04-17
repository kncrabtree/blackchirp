#include "led.h"
#include <QPainter>
#include <QPalette>

Led::Led(QWidget *parent) : QWidget(parent), d_ledOn(false)
{
    setLedSize(15);
    d_onColor = QColor(0,175,0);
    d_offColor = QPalette().color(QPalette::Base);
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

//    QRadialGradient g(width()/2,height()/2,d_diameter/2,d_diameter/4,d_diameter/4);


    if(d_ledOn)
        p.setBrush(QBrush(d_onColor));
    else
        p.setBrush(QBrush(d_offColor));

    p.drawEllipse(0,0,d_diameter,d_diameter);

}
