#ifndef HWDIALOG_H
#define HWDIALOG_H

#include <QDialog>

class HwSettingsWidget;

class HWDialog : public QDialog
{
    Q_OBJECT
public:
    HWDialog(const QString &key, QWidget *controlWidget = nullptr,
             QWidget *managedWidget = nullptr, QWidget *parent = nullptr);

    void discardControlWidget();

    QString hwKey() const { return d_hwKey; }

private:
    HwSettingsWidget *p_settingsWidget;
    QString d_hwKey;
    QWidget *p_controlWidget;
    QWidget *p_managedWidget; // sub-widget toggled by setControlWidgetEnabled (defaults to p_controlWidget)

public slots:
    void accept() override;
    QSize sizeHint() const override;

    // Enables or disables the per-device control surface. For composite control
    // widgets (e.g., the Python wrapper), only the inner managed widget is
    // toggled so that always-on controls (script reload, etc.) remain reachable.
    void setControlWidgetEnabled(bool enabled);
};

#endif // HWDIALOG_H
