#ifndef WIZARDVALIDATIONPAGE_H
#define WIZARDVALIDATIONPAGE_H

#include <gui/wizard/experimentwizardpage.h>

#include <QTableView>

#include <data/experiment/ioboardconfig.h>

class QToolButton;

namespace BC::Key::WizVal {
static const QString key("WizardValidationPage");
}

class WizardValidationPage : public ExperimentWizardPage
{
public:
    explicit WizardValidationPage(QWidget *parent = nullptr);

private:
    QTableView *p_analogView, *p_digitalView, *p_validationView;
    QToolButton *p_addButton, *p_removeButton;

    // QWizardPage interface
public:
    int nextId() const;
    virtual void initializePage();
    virtual bool validatePage();

    IOBoardConfig getConfig() const;
    QMap<QString,BlackChirp::ValidationItem> getValidation() const;

};

#endif // WIZARDVALIDATIONPAGE_H
