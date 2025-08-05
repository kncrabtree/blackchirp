#ifndef LIFCONTROLWIDGET_H
#define LIFCONTROLWIDGET_H

#include <QWidget>

#include <data/storage/settingsstorage.h>

#include <data/lif/liftrace.h>
#include <data/lif/lifconfig.h>
#include <data/lif/lifdigitizerconfig.h>

class LifTracePlot;
class DigitizerConfigWidget;
class LifLaserWidget;
class LifProcessingWidget;
class QPushButton;
class QSpinBox;

namespace BC::Key::LifControl {
const QString key("lifControlWidget");
const QString avgs("numAverages");
const QString lifDigWidget("lifDigitizerConfig");
}

class LifControlWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT

public:
    explicit LifControlWidget(const QString& scopeHwType, const QString& scopeImpl, const QString& scopeLabel, QWidget *parent = nullptr);
    explicit LifControlWidget(const LifConfig& config, QWidget *parent = nullptr);
    ~LifControlWidget() override;

    void startAcquisition();
    void stopAcquisition();
    void acquisitionStarted();
    void newWaveform(const QVector<qint8> b);

    void setLaserPosition(const double d);
    void setFlashlamp(bool en);

    void setFromConfig(const LifConfig &cfg);
    void toConfig(LifConfig &cfg);

signals:
    void startSignal(LifConfig);
    void stopSignal();
    void changeLaserPosSignal(double);
    void changeLaserFlashlampSignal(bool);

private:
    void initializeWidget(); // Common initialization for both constructors
    
    LifTracePlot *p_lifTracePlot;
    DigitizerConfigWidget *p_digWidget;
    LifLaserWidget *p_laserWidget;
    LifProcessingWidget *p_procWidget;

    QPushButton *p_startAcqButton;
    QPushButton *p_stopAcqButton;
    QSpinBox *p_avgBox;
    QPushButton *p_resetButton;

    LifConfig d_cfg;
    bool d_acquiring{ false };


    // QWidget interface
public:
    QSize sizeHint() const override;
};

#endif // LIFCONTROLWIDGET_H
