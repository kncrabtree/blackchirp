#ifndef HWDIALOG_H
#define HWDIALOG_H

#include <QDialog>

class HwSettingsWidget;
class QLabel;
class QPushButton;

class HWDialog : public QDialog
{
    Q_OBJECT
public:
    HWDialog(const QString &key, QWidget *controlWidget = nullptr,
             QWidget *managedWidget = nullptr, QWidget *parent = nullptr);

    void discardControlWidget();

    QString hwKey() const { return d_hwKey; }

signals:
    void requestTestConnection(const QString &hwKey);
    void requestCommunicationDialog(const QString &hwKey);

public slots:
    void accept() override;
    QSize sizeHint() const override;

    // Enables or disables the per-device control surface. For composite control
    // widgets (e.g., the Python wrapper), only the inner managed widget is
    // toggled so that always-on controls (script reload, etc.) remain reachable.
    void setControlWidgetEnabled(bool enabled);

    void setConnectionStatus(bool connected);

    void onConnectionResult(const QString &hwKey, bool success, const QString &msg);

private:
    HwSettingsWidget *p_settingsWidget;
    QString d_hwKey;
    QWidget *p_controlWidget;
    QWidget *p_managedWidget; // sub-widget toggled by setControlWidgetEnabled (defaults to p_controlWidget)
    QPushButton *p_testButton;
    QLabel *p_statusLabel;
    bool d_testInFlight{false};
};

#endif // HWDIALOG_H
