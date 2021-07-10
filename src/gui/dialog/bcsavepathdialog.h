#ifndef BCSAVEPATHDIALOG_H
#define BCSAVEPATHDIALOG_H

#include <QDialog>
#include <data/storage/settingsstorage.h>

class QSpinBox;
class QDialogButtonBox;
class QLineEdit;

class BCSavePathDialog : public QDialog, public SettingsStorage
{
public:
    BCSavePathDialog();
    QSize sizeHint() const override;

private:
    QSpinBox *p_expBox;
    QDialogButtonBox *p_buttons;
    QLineEdit *p_lineEdit;

public slots:
    void accept() override;
    void reject() override;
    void apply();


};

#endif // BCSAVEPATHDIALOG_H
