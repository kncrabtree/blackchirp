#ifndef LIFCONTROLWIDGET_H
#define LIFCONTROLWIDGET_H

#include <QWidget>

#include <src/data/datastructs.h>
#include <src/modules/lif/data/liftrace.h>
#include <src/modules/lif/data/lifconfig.h>

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
    BlackChirp::LifScopeConfig toConfig() const;

signals:
    void updateScope(BlackChirp::LifScopeConfig);
    void newTrace(LifTrace);
    void lifColorChanged();
    void laserPosUpdate(double pos);

public slots:
    void scopeConfigChanged(BlackChirp::LifScopeConfig c);
    void checkLifColors();
    void updateHardwareLimits();
    void setLaserPos(double pos);
    void setSampleRateBox(double rate);

private:
    Ui::LifControlWidget *ui;

};

#endif // LIFCONTROLWIDGET_H
