#ifndef ADDPROFILEDIALOG_H
#define ADDPROFILEDIALOG_H

#include <QDialog>

class QComboBox;
class QLineEdit;
class QLabel;
class QWidget;
class QDialogButtonBox;
class HwSettingsWidget;

class AddProfileDialog : public QDialog
{
    Q_OBJECT
public:
    explicit AddProfileDialog(const QString &hardwareType, QWidget *parent = nullptr);

    QString selectedImplementation() const;
    QString profileLabel() const;
    QString pythonScriptPath() const;

private:
    void updateProtocolCombo(const QString &impl);
    void updateSettingsDefs(const QString &impl);
    void validateLabel(const QString &text);
    void offerPythonTemplate();

    void accept() override;

    QString d_hardwareType;
    QString d_pythonScriptPath;

    QComboBox *p_implementationCombo;
    QComboBox *p_protocolCombo;
    QLabel *p_protocolLabel;
    QLineEdit *p_labelEdit;
    QLabel *p_validationLabel;
    QWidget *p_settingsContainer;
    HwSettingsWidget *p_settingsWidget{nullptr};
    QDialogButtonBox *p_buttonBox;
};

#endif // ADDPROFILEDIALOG_H
