#ifndef FTMWPROCESSINGPANEL_H
#define FTMWPROCESSINGPANEL_H

#include <QWidget>

#include <data/storage/settingsstorage.h>
#include <data/experiment/experiment.h>
#include <data/analysis/ftworker.h>

class QSpinBox;
class QDoubleSpinBox;
class QComboBox;
class QCheckBox;
class QPushButton;
class QTableWidget;

namespace BC::Key {
inline constexpr QLatin1StringView ftmwProcWidget{"ftmwProcessingWidget"};
inline constexpr QLatin1StringView fidStart{"startUs"};
inline constexpr QLatin1StringView fidEnd{"endUs"};
inline constexpr QLatin1StringView fidExp{"expfUs"};
inline constexpr QLatin1StringView zeroPad{"zeroPad"};
inline constexpr QLatin1StringView removeDC{"removeDC"};
inline constexpr QLatin1StringView ftUnits{"ftUnits"};
inline constexpr QLatin1StringView autoscaleIgnore{"autoscaleIgnoreMHz"};
inline constexpr QLatin1StringView ftWinf{"windowFunction"};
}

class FtmwProcessingPanel : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    explicit FtmwProcessingPanel(bool mainWin = false, QWidget *parent = nullptr);
    ~FtmwProcessingPanel();
    FtWorker::FidProcessingSettings getSettings();

signals:
    void settingsUpdated(FtWorker::FidProcessingSettings);
    void resetSignal();
    void saveSignal();

public slots:
    void setAll(const FtWorker::FidProcessingSettings &c);
    void prepareForExperient(const Experiment &e);
    void readSettings();

private:
    QTableWidget *p_table;
    QDoubleSpinBox *p_startBox, *p_endBox, *p_expBox, *p_autoScaleIgnoreBox;
    QSpinBox *p_zeroPadBox;
    QCheckBox *p_removeDCBox;
    QComboBox *p_unitsBox, *p_winfBox;
    QPushButton *p_resetButton, *p_saveButton;

    int valueToWinfIndex(FtWorker::FtWindowFunction w) const;
    int valueToUnitsIndex(FtWorker::FtUnits u) const;
};

#endif // FTMWPROCESSINGPANEL_H
