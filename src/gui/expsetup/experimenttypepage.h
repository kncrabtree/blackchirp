#ifndef EXPERIMENTTYPEPAGE_H
#define EXPERIMENTTYPEPAGE_H

#include "experimentconfigpage.h"
#include <data/experiment/ftmwconfig.h>
#include "loscanconfigwidget.h"
#include "drscanconfigwidget.h"

class QGroupBox;
class QSpinBox;
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QStackedWidget;

namespace BC::Key::WizStart {
inline constexpr QLatin1StringView key{"ExperimentTypePage"};
inline constexpr QLatin1StringView title{"Experiment Type"};
inline constexpr QLatin1StringView ftmw{"FtmwEnabled"};
inline constexpr QLatin1StringView ftmwType{"FtmwType"};
inline constexpr QLatin1StringView ftmwShots{"FtmwShots"};
inline constexpr QLatin1StringView ftmwDuration{"FtmwDuration"};
inline constexpr QLatin1StringView ftmwPhase{"FtmwPhaseCorrection"};
inline constexpr QLatin1StringView ftmwScoring{"FtmwChirpScoring"};
inline constexpr QLatin1StringView ftmwThresh{"FtmwChirpScoringThreshold"};
inline constexpr QLatin1StringView ftmwOffset{"FtmwChirpOffset"};
inline constexpr QLatin1StringView auxInterval{"AuxDataInterval"};
inline constexpr QLatin1StringView backup{"BackupInterval"};
inline constexpr QLatin1StringView lif{"LifEnabled"};
inline constexpr QLatin1StringView lifDelayStart{"LifDelayStart"};
inline constexpr QLatin1StringView lifDelayStep{"LifDelayStep"};
inline constexpr QLatin1StringView lifDelayPoints{"LifDelayPoints"};
inline constexpr QLatin1StringView lifDelayRandom{"LifDelayRandom"};
inline constexpr QLatin1StringView lifLaserStart{"LifLaserStart"};
inline constexpr QLatin1StringView lifLaserStep{"LifLaserStep"};
inline constexpr QLatin1StringView lifLaserPoints{"LifLaserPoints"};
inline constexpr QLatin1StringView lifOrder{"LifOrder"};
inline constexpr QLatin1StringView lifCompleteMode{"LifCompleteMode"};
inline constexpr QLatin1StringView lifFlashlampDisable{"LifFlashlampDisable"};
}

class ExperimentTypePage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    ExperimentTypePage(Experiment *exp, QWidget *parent = nullptr);

    bool ftmwEnabled() const ;
    FtmwConfig::FtmwType getFtmwType() const;
    bool lifEnabled() const;

signals:
    void typeChanged();

    // ExperimentConfigPage interface
public slots:
    void initialize() override;
    bool validate() override;
    void apply() override;
    void configureUI();

private:
    QGroupBox *p_ftmw;
    QStackedWidget *p_ftmwConfigStack;

    QGroupBox *p_lif;
    QDoubleSpinBox *p_dStartBox, *p_dStepBox, *p_dEndBox, *p_lStartBox, *p_lStepBox, *p_lEndBox;
    QSpinBox *p_dNumStepsBox, *p_lNumStepsBox;
    QComboBox *p_orderBox, *p_completeModeBox;
    QCheckBox *p_flBox, *p_delayRandomBox;

    void updateLifRanges();

    QSpinBox *p_auxDataIntervalBox, *p_backupBox, *p_ftmwShotsBox, *p_ftmwTargetDurationBox;
    QWidget *p_ftmwShotsWidget, *p_ftmwTargetDurationWidget, *p_foreverWidget;
    LOScanConfigWidget *p_loScanConfigWidget;
    DRScanConfigWidget *p_drScanConfigWidget;
    QComboBox *p_ftmwTypeBox;
    QCheckBox *p_phaseCorrectionBox, *p_chirpScoringBox;
    QDoubleSpinBox *p_thresholdBox, *p_chirpOffsetBox;
    QLabel *p_endTimeLabel;
    int d_timerId;

private slots:
    void updateLabel();


    // QObject interface
protected:
    void timerEvent(QTimerEvent *event) override;
};

#endif // EXPERIMENTTYPEPAGE_H
