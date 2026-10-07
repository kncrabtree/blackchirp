#ifndef FCUTUNEDIALOG_H
#define FCUTUNEDIALOG_H

#include <functional>

#include <QDialog>
#include <QVector>
#include <QPointF>

#include <data/storage/settingsstorage.h>
#include <data/lif/fcutunesweep.h>

class FcuTuneController;
class LifConfig;
class LifSlicePlot;
class QComboBox;
class QSpinBox;
class QDoubleSpinBox;
class QPushButton;
class QProgressBar;
class QLabel;

namespace BC::Key::FcuTuneDialog {
inline constexpr QLatin1StringView key{"fcuTuneDialog"};
inline constexpr QLatin1StringView halfWidth{"halfWidth"};
inline constexpr QLatin1StringView points{"points"};
inline constexpr QLatin1StringView waveforms{"waveformsPerPoint"};
inline constexpr QLatin1StringView discard{"discardPerPoint"};
inline constexpr QLatin1StringView minContrast{"minContrastPercent"};
inline constexpr QLatin1StringView saturation{"saturationVolts"};
inline constexpr QLatin1StringView recenters{"maxRecenters"};
inline constexpr QLatin1StringView plot{"FcuTunePlot"};
}

/*!
 * \brief Non-modal dialog for on-demand phase-match trim sweeps of a LIF
 *        frequency-conversion stage, and for setting its trim by hand.
 *
 * Owned by LifControlWidget, which forwards LIF waveforms, the acquisition
 * state, and HardwareManager's trim reports, and relays this dialog's trim
 * requests. A sweep needs a running LIF acquisition with the reference
 * channel enabled and its gate set; \a configProvider fills a LifConfig with
 * the digitizer configuration and processing settings in use.
 */
class FcuTuneDialog : public QDialog, public SettingsStorage
{
    Q_OBJECT
public:
    FcuTuneDialog(const QString &digitizerHwKey, std::function<void(LifConfig&)> configProvider,
                  QWidget *parent = nullptr);
    ~FcuTuneDialog() override;

    void setAcquiring(bool acquiring);

public slots:
    void stageTrimUpdate(const QString &stageKey, double trim, int direction, bool success);
    void newWaveform(const QVector<qint8> b);

signals:
    void requestTrim(QString stageKey, double trim);
    void requestTrimReport();

protected:
    void showEvent(QShowEvent *e) override;
    void closeEvent(QCloseEvent *e) override;
    void reject() override;

private:
    void startSweep();
    void pointComplete(double trim, double mean, double stdErr);
    void sweepFinished(const BC::FcuTune::Result &r, double finalTrim, bool moveOk);
    void updateControls();
    void updateTrimLabel();

    QString d_digitizerHwKey;
    std::function<void(LifConfig&)> d_configProvider;
    FcuTuneController *p_controller;
    bool d_acquiring{false};

    struct StageInfo {
        double trim{0.0};
        int direction{1};
    };
    std::map<QString,StageInfo> d_stages;
    QVector<QPointF> d_points;

    QComboBox *p_stageBox;
    QLabel *p_trimLabel;
    QSpinBox *p_halfWidthBox;
    QSpinBox *p_pointsBox;
    QSpinBox *p_waveformsBox;
    QSpinBox *p_discardBox;
    QDoubleSpinBox *p_contrastBox;
    QDoubleSpinBox *p_saturationBox;
    QSpinBox *p_recentersBox;
    QSpinBox *p_manualTrimBox;
    QPushButton *p_setTrimButton;
    QPushButton *p_zeroTrimButton;
    QPushButton *p_startButton;
    QPushButton *p_abortButton;
    QProgressBar *p_progressBar;
    QLabel *p_statusLabel;
    LifSlicePlot *p_plot;
};

#endif // FCUTUNEDIALOG_H
