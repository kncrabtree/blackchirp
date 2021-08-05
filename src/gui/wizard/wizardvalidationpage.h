#ifndef WIZARDVALIDATIONPAGE_H
#define WIZARDVALIDATIONPAGE_H

#include <gui/wizard/experimentwizardpage.h>

#include <QTableView>

#include <hardware/optional/ioboard/ioboardconfig.h>

class QToolButton;

namespace BC::Key::WizVal {
static const QString key("WizardValidationPage");
}

class WizardValidationPage : public ExperimentWizardPage
{
public:
    explicit WizardValidationPage(QWidget *parent = nullptr);

    void setValidationKeys(std::map<QString,QStringList> m);

private:
    QTableView *p_validationView;
    QToolButton *p_addButton, *p_removeButton;

    std::map<QString,QStringList> d_validationKeys;

    // QWizardPage interface
public:
    int nextId() const;
    virtual void initializePage();
    virtual bool validatePage();

};

#endif // WIZARDVALIDATIONPAGE_H
