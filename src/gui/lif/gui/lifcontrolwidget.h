#ifndef LIFCONTROLWIDGET_H
#define LIFCONTROLWIDGET_H

#include <QWidget>

#include <memory>

#include <data/storage/settingsstorage.h>

#include <data/lif/liftrace.h>
#include <data/lif/lifconfig.h>
#include <data/lif/lifdigitizerconfig.h>

class LifTracePlot;
class DigitizerConfigWidget;
class LifLaserWidget;
class LifProcessingWidget;
class QToolButton;
class QSpinBox;
class FcuTuneDialog;

namespace BC::Key::LifControl {
const QString key("lifControlWidget");
const QString avgs("numAverages");
const QString lifDigWidget("lifDigitizerConfig");
}

class LifControlWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT

public:
    explicit LifControlWidget(const QString& digitizerHwKey, const QString& laserHwKey, QWidget *parent = nullptr);
    ~LifControlWidget() override;

    void startAcquisition();
    void stopAcquisition();
    void acquisitionStarted();
    void newWaveform(const QVector<qint8> b);

    void setLaserPosition(const double d);
    void setFlashlamp(bool en);
    void stageTrimUpdate(const QString &stageKey, double trim, int direction, bool success);

    void setFromConfig(const LifConfig &cfg);
    void toConfig(LifConfig &cfg);

signals:
    void startSignal(LifConfig);
    void stopSignal();
    void changeLaserPosSignal(double);
    void changeLaserFlashlampSignal(bool);
    void changeStageTrimSignal(QString stageKey, double trim);
    void requestStageTrimReport();

private:
    void initializeWidget();
    void showFcuTuneDialog();
    
    LifTracePlot *p_lifTracePlot;
    DigitizerConfigWidget *p_digWidget;
    LifLaserWidget *p_laserWidget;
    LifProcessingWidget *p_procWidget;

    QToolButton *p_startAcqButton;
    QToolButton *p_stopAcqButton;
    QSpinBox *p_avgBox;
    QToolButton *p_resetButton;
    QToolButton *p_fcuTuneButton;
    FcuTuneDialog *p_fcuTuneDialog{nullptr};

    std::shared_ptr<LifConfig> ps_cfg;
    QString d_laserHwKey;
    QString d_digitizerHwKey;
    bool d_acquiring{ false };
};

#endif // LIFCONTROLWIDGET_H
