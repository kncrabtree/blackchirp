#ifndef WIZARDSTARTPAGE_H
#define WIZARDSTARTPAGE_H

#include <gui/wizard/experimentwizardpage.h>

class QGroupBox;
class QSpinBox;
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;

namespace BC::Key::WizStart {
static const QString key{"WizardStartPage"};
static const QString ftmw{"FtmwEnabled"};
static const QString ftmwType{"FtmwType"};
static const QString ftmwShots("FtmwShots");
static const QString ftmwDuration("FtmwDuration");
static const QString ftmwPhase("FtmwPhaseCorrection");
static const QString ftmwScoring("FtmwChirpScoring");
static const QString ftmwThresh("FtmwChirpScoringThreshold");
static const QString ftmwOffset("FtmwChirpOffset");
static const QString auxInterval("AuxDataInterval");
static const QString autosave("AutosaveInterval");
}

class WizardStartPage : public ExperimentWizardPage
{
    Q_OBJECT
public:
    WizardStartPage(QWidget *parent = 0);
    ~WizardStartPage();

    // QWizardPage interface
    int nextId() const override;
    bool isComplete() const override;
    void initializePage() override;
    bool validatePage() override;

public slots:
    void configureUI();

private:
    QGroupBox *p_ftmw;
#ifdef BC_LIF
    QGroupBox *p_lif;
#endif
#ifdef BC_MOTOR
    QGroupBox *p_motor;
#endif

    QSpinBox *p_auxDataIntervalBox, *p_autosaveBox, *p_ftmwShotsBox, *p_ftmwTargetDurationBox;
    QComboBox *p_ftmwTypeBox;
    QCheckBox *p_phaseCorrectionBox, *p_chirpScoringBox;
    QDoubleSpinBox *p_thresholdBox, *p_chirpOffsetBox;
    QLabel *p_endTimeLabel;
    int d_timerId;
    QString d_endText{"Est. end: %1"};




    // QObject interface
protected:
    void timerEvent(QTimerEvent *event) override;
};

#endif // WIZARDSTARTPAGE_H
