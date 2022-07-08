#ifndef LIFCONTROLWIDGET_H
#define LIFCONTROLWIDGET_H

#include <QWidget>

#include <modules/lif/data/liftrace.h>
#include <modules/lif/data/lifconfig.h>

class LifTracePlot;
class DigitizerConfigWidget;
class LifLaserWidget;
class LifProcessingWidget;
class QPushButton;

namespace BC::Key::LifControl {
const QString lifDigWidget("lifDigitizerConfig");
}

class LifControlWidget : public QWidget
{
    Q_OBJECT

public:
    explicit LifControlWidget(QWidget *parent = nullptr);
    ~LifControlWidget() override;

private:
    LifTracePlot *p_lifTracePlot;
    DigitizerConfigWidget *p_digWidget;
    LifLaserWidget *p_laserWidget;
    LifProcessingWidget *p_procWidget;

    QPushButton *p_startAcqButton;
    QPushButton *p_stopAcqButton;


    // QWidget interface
public:
    QSize sizeHint() const override;
};

#endif // LIFCONTROLWIDGET_H
