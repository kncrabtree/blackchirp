#ifndef EXPERIMENTVALIDATORCONFIGPAGE_H
#define EXPERIMENTVALIDATORCONFIGPAGE_H

#include "experimentconfigpage.h"

class QTableView;
class QToolButton;

namespace BC::Key::WizardVal {
static const QString key{"ValidationPage"};
static const QString title{"Validation Settings"};
}

class ExperimentValidatorConfigPage : public ExperimentConfigPage
{
    Q_OBJECT
public:
    ExperimentValidatorConfigPage(Experiment *exp, const std::map<QString, QStringList> &valKeys, QWidget *parent = nullptr);

private:
    QTableView *p_validationView;
    QToolButton *p_addButton, *p_removeButton;

    // ExperimentConfigPage interface
public slots:
    void initialize() override;
    bool validate() override;
    void apply() override;
};

#endif // EXPERIMENTVALIDATORCONFIGPAGE_H
