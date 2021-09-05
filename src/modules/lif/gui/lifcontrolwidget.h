#ifndef LIFCONTROLWIDGET_H
#define LIFCONTROLWIDGET_H

#include <QWidget>

#include <modules/lif/data/liftrace.h>
#include <modules/lif/data/lifconfig.h>

namespace Ui {
class LifControlWidget;
}

class LifControlWidget : public QWidget
{
    Q_OBJECT

public:
    explicit LifControlWidget(QWidget *parent = nullptr);
    ~LifControlWidget() override;

    LifConfig getSettings(LifConfig c);
    double laserPos() const;
    Blackchirp::LifScopeConfig toConfig() const;

signals:
    void updateScope(Blackchirp::LifScopeConfig);
    void newTrace(LifTrace);
    void laserPosUpdate(double pos);

public slots:
    void scopeConfigChanged(Blackchirp::LifScopeConfig c);
    void checkLifColors();
    void updateHardwareLimits();
    void setLaserPos(double pos);
    void setSampleRateBox(double rate);

private:
    Ui::LifControlWidget *ui;

};

#endif // LIFCONTROLWIDGET_H
