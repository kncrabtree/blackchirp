#ifndef EXPERIMENTTYPEPAGE_H
#define EXPERIMENTTYPEPAGE_H

#include "experimentconfigpage.h"
#include <data/experiment/ftmwconfig.h>

class QGroupBox;
class QSpinBox;
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;

namespace BC::Key::WizStart {
static const QString key{"ExperimentTypePage"};
static const QString title{"Experiment Type"};
static const QString ftmw{"FtmwEnabled"};
static const QString ftmwType{"FtmwType"};
static const QString ftmwShots{"FtmwShots"};
static const QString ftmwDuration{"FtmwDuration"};
static const QString ftmwPhase{"FtmwPhaseCorrection"};
static const QString ftmwScoring{"FtmwChirpScoring"};
static const QString ftmwThresh{"FtmwChirpScoringThreshold"};
static const QString ftmwOffset{"FtmwChirpOffset"};
static const QString auxInterval{"AuxDataInterval"};
static const QString backup{"BackupInterval"};
#ifdef BC_LIF
static const QString lif{"LifEnabled"};
static const QString lifDelayStart{"LifDelayStart"};
static const QString lifDelayStep{"LifDelayStep"};
static const QString lifDelayPoints{"LifDelayPoints"};
static const QString lifLaserStart{"LifLaserStart"};
static const QString lifLaserStep{"LifLaserStep"};
static const QString lifLaserPoints{"LifLaserPoints"};
static const QString lifOrder{"LifOrder"};
static const QString lifCompleteMode{"LifCompleteMode"};
static const QString lifFlashlampDisable{"LifFlashlampDisable"};
#endif
}

class ExperimentTypePage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    ExperimentTypePage(Experiment *exp, QWidget *parent = nullptr);

    bool ftmwEnabled() const ;
    FtmwConfig::FtmwType getFtmwType() const;
#ifdef BC_LIF
    bool lifEnabled() const;
#endif

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
#ifdef BC_LIF
    QGroupBox *p_lif;
    QDoubleSpinBox *p_dStartBox, *p_dStepBox, *p_dEndBox, *p_lStartBox, *p_lStepBox, *p_lEndBox;
    QSpinBox *p_dNumStepsBox, *p_lNumStepsBox;
    QComboBox *p_orderBox, *p_completeModeBox;
    QCheckBox *p_flBox;

    void updateLifRanges();
#endif

    QSpinBox *p_auxDataIntervalBox, *p_backupBox, *p_ftmwShotsBox, *p_ftmwTargetDurationBox;
    QComboBox *p_ftmwTypeBox;
    QCheckBox *p_phaseCorrectionBox, *p_chirpScoringBox;
    QDoubleSpinBox *p_thresholdBox, *p_chirpOffsetBox;
    QLabel *p_endTimeLabel;
    int d_timerId;
    QString d_endText{"Est. end: %1"};

private slots:
    void updateLabel();


    // QObject interface
protected:
    void timerEvent(QTimerEvent *event) override;
};

#endif // EXPERIMENTTYPEPAGE_H
