#ifndef EXPERIMENTWIZARD_H
#define EXPERIMENTWIZARD_H

#include <QWizard>
#include <memory>

#include <data/experiment/experiment.h>

class BatchManager;
class ExperimentWizardPage;

class ExperimentWizard : public QWizard
{
    Q_OBJECT
public:
    ExperimentWizard(Experiment *exp, std::map<QString, QString> hw, QWidget *parent = 0);
    ~ExperimentWizard();

    enum Page {
        StartPage=1,
        LoScanPage,
        DrScanPage,
        RfConfigPage,
        ChirpConfigPage,
        DigitizerConfigPage,
#ifdef BC_LIF
        LifConfigPage,
#endif
        PulseConfigPage,
        IOBoardConfigPage,
        ValidationPage,
        SummaryPage
    };

    Experiment* p_experiment;
    void setValidationKeys(const std::map<QString,QStringList> &m);
    QHash<RfConfig::ClockType, RfConfig::ClockFreq> d_clocks;
    Page nextOptionalPage();


private:
    QVector<Page> d_optionalPages;

#ifdef BC_LIF
public slots:
    void setCurrentLaserPos(double pos);

signals:
    void newTrace(LifTrace);
    void updateScope(BlackChirp::LifScopeConfig);
    void scopeConfigChanged(BlackChirp::LifScopeConfig);
    void lifColorChanged();
    void laserPosUpdate(double);

private:
    ExperimentWizardPage *p_lifConfigPage;
#endif

#ifdef BC_MOTOR
private:
    ExperimentWizardPage *p_motorScanConfigPage;
#endif


    // QWidget interface
public:
    virtual QSize sizeHint() const override;

    // QDialog interface
public slots:
    void reject() override;
};

#endif // EXPERIMENTWIZARD_H
