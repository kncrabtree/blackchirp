#ifndef WIZARDSUMMARYPAGE_H
#define WIZARDSUMMARYPAGE_H

#include <QWizardPage>

class QPlainTextEdit;

class WizardSummaryPage : public QWizardPage
{
    Q_OBJECT
public:
    WizardSummaryPage(QWidget *parent = 0);
    ~WizardSummaryPage();

    // QWizardPage interface
    void initializePage();
    int nextId() const;

private:
    QPlainTextEdit *p_pte;
};

#endif // WIZARDSUMMARYPAGE_H
