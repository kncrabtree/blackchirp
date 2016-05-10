#ifndef LIFCONTROLWIDGET_H
#define LIFCONTROLWIDGET_H

#include <QWidget>

#include "datastructs.h"
#include "liftrace.h"
#include "lifconfig.h"

namespace Ui {
class LifControlWidget;
}

class LifControlWidget : public QWidget
{
    Q_OBJECT

public:
    explicit LifControlWidget(QWidget *parent = 0);
    ~LifControlWidget();

    LifConfig getSettings(LifConfig c);
    BlackChirp::LifScopeConfig toConfig() const;

signals:
    void updateScope(const BlackChirp::LifScopeConfig);
    void newTrace(const LifTrace);
    void lifColorChanged();

public slots:
    void scopeConfigChanged(const BlackChirp::LifScopeConfig c);
    void checkLifColors();
    void updateHardwareLimits();

private:
    Ui::LifControlWidget *ui;

};

#endif // LIFCONTROLWIDGET_H
