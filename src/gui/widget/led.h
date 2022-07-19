#ifndef LED_H
#define LED_H

#include <QWidget>
#include <QColor>

class Led : public QWidget
{
    Q_OBJECT
public:
    enum LedColor{
        Red = 0xb00000,
        Orange = 0xb06000,
        Yellow = 0xb0b000,
        Green = 0x00b000,
        Blue = 0x0000b0,
        Purple = 0xb000b0
    };
    Q_ENUM(LedColor)

    explicit Led(QWidget *parent = 0);
    explicit Led(LedColor color, int size = 15, QWidget *parent = 0);
    ~Led();


signals:

public slots:
    void setLedSize(int newSize);
    void setState(bool on);
    void setColor(LedColor c);

    // QWidget interface
protected:
    void paintEvent(QPaintEvent *ev);

private:
    LedColor d_color{Green};
    int d_diameter{15};
    bool d_ledOn{false};
};

#endif // LED_H
