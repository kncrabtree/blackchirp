#ifndef LED_H
#define LED_H

#include <QWidget>
#include <QColor>

class Led : public QWidget
{
    Q_OBJECT
public:
    explicit Led(QWidget *parent = 0);
    ~Led();

signals:

public slots:
    void setLedSize(int newSize);
    void setState(bool on);

    // QWidget interface
protected:
    void paintEvent(QPaintEvent *ev);

private:
    QColor d_onColor, d_offColor;
    int d_diameter;
    bool d_ledOn;
};

#endif // LED_H
