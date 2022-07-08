#ifndef LIFLASERWIDGET_H
#define LIFLASERWIDGET_H

#include <QWidget>

class QDoubleSpinBox;
class QPushButton;

class LifLaserWidget : public QWidget
{
    Q_OBJECT
public:
    explicit LifLaserWidget(QWidget *parent = nullptr);

    void setPosition(const double d);
    void setFlashlamp(bool b);

signals:
    void changePosition(double);
    void changeFlashlamp(bool);

private:
    QDoubleSpinBox *p_posBox;
    QPushButton *p_posSetButton;
    QPushButton *p_flButton;
};

#endif // LIFLASERWIDGET_H
