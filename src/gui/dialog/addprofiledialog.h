#ifndef ADDPROFILEDIALOG_H
#define ADDPROFILEDIALOG_H

#include <QDialog>
#include <QHash>

class QComboBox;
class QLineEdit;
class QLabel;
class QGroupBox;
class QFormLayout;
class QDialogButtonBox;

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
    void updateConfigParams(const QString &impl);
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
    QGroupBox *p_configParamsGroup;
    QFormLayout *p_configParamsLayout;
    QDialogButtonBox *p_buttonBox;
    QHash<QString, QWidget*> d_paramWidgets;
};

#endif // ADDPROFILEDIALOG_H
