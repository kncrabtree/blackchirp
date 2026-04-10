#ifndef PYTHONSETTINGSWIDGET_H
#define PYTHONSETTINGSWIDGET_H

#include <QWidget>

class QLineEdit;
class QComboBox;
class QLabel;
class QTimer;

class PythonSettingsWidget : public QWidget
{
    Q_OBJECT
public:
    explicit PythonSettingsWidget(QWidget *parent = nullptr);

    QString scriptPath() const;
    void setScriptPath(const QString &path);

    QString className() const;
    void setClassName(const QString &name);
    void setClassNamePlaceholder(const QString &placeholder);

    QString envPath() const;
    void setEnvPath(const QString &path);

signals:
    void scriptPathChanged(const QString &path);
    void classNameChanged(const QString &name);
    void envPathChanged(const QString &path);

private:
    void populateClassCombo(const QString &scriptPath);
    void updateEnvStatus();
    static QString getPythonVersion(const QString &exe);

    QLineEdit *p_scriptEdit;
    QComboBox *p_classCombo;
    QLineEdit *p_envEdit;
    QLabel *p_envStatusLabel;
    QTimer *p_envStatusTimer;
};

#endif // PYTHONSETTINGSWIDGET_H
