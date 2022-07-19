#include <gui/widget/led.h>

#include <QPainter>
#include <QPalette>
#include <QRadialGradient>

Led::Led(QWidget *parent) : QWidget(parent)
{
    setLedSize(d_diameter);
}

Led::Led(LedColor color, int size, QWidget *parent) : QWidget(parent), d_color{color}
{
    setLedSize(size);
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

void Led::setColor(LedColor c)
{
    d_color = c;
    update();
}



void Led::paintEvent(QPaintEvent *ev)
{
    Q_UNUSED(ev)

    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    int offset = (width()-d_diameter)/2;

    QRadialGradient g(offset+d_diameter/2,offset+d_diameter/2,35*d_diameter/100,2*d_diameter/5,2*d_diameter/5);

    int rr = (d_color & 0xff0000) >> 16;
    int gg = (d_color & 0x00ff00) >> 8;
    int bb = (d_color & 0x0000ff);

    QColor on(rr,gg,bb);
    QColor off(rr/5,gg/5,bb/5);

    if(d_ledOn)
    {
        g.setColorAt(0,QColor(255,255,255,200));
        g.setColorAt(1,on);
        p.setPen(on);
    }
    else
    {
        g.setColorAt(0,QColor(220,220,220,150));
        g.setColorAt(1,off);
        p.setPen(off);
    }

    p.setBrush(QBrush(g));
    p.drawEllipse(QPoint(width()/2,height()/2),d_diameter/2,d_diameter/2);

}

