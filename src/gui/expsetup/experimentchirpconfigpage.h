#ifndef EXPERIMENTCHIRPCONFIGPAGE_H
#define EXPERIMENTCHIRPCONFIGPAGE_H

#include "experimentconfigpage.h"

class ChirpConfigWidget;
class RfConfig;

namespace BC::Key::WizChirp {
static const QString key{"WizardChirpConfigPage"};
static const QString title{"Chirp(s)"};
}

class ExperimentChirpConfigPage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    ExperimentChirpConfigPage(Experiment *exp, QWidget *parent = nullptr);

    // ExperimentConfigPage interface
public slots:
    void initialize() override;
    bool validate() override;
    void apply() override;

private:
    ChirpConfigWidget *p_ccw;
};

#endif // EXPERIMENTCHIRPCONFIGPAGE_H
